"""Shadow tracking for the candidates the validation gate *rejected*.

The promotion FSM measures forward performance for strategies that passed the gate. The
ones it rejected are written to the population and never looked at again, which leaves the
most useful question about the gate unanswerable: **is it rejecting noise, or is it
rejecting edge?**

A gate is a classifier, and a classifier with no measured false-negative rate is a
classifier nobody has checked. Every threshold in it — ``dsr_min``, ``pbo_max``,
``spa_max`` — is a guess about where noise stops and signal starts. If a third of the
rejects go on to earn a positive forward Sharpe at the same band a promoted strategy must
clear, the thresholds are throwing away real strategies. If almost none do, the gate is
doing its job and the shortage of champions is a shortage of edge. Those two diagnoses
call for opposite actions, and until the rejects are tracked there is no way to tell them
apart.

This module answers it for free. Nothing here places an order or allocates capital: it
recomputes returns from bars already in the store, exactly as ``promotion.forward_returns``
does, and judges them with exactly the same test a promoted strategy faces
(``promotion.incubation_passed``). Using a different measuring stick for the two groups
would make the comparison worthless, so the test is imported rather than reimplemented.

**Tracks are immutable.** A rejected family is re-searched every night and its population
row is overwritten with whatever the search lands on, so the row itself is a moving target
— measuring "the candidate" against a row that changes nightly would reproduce the very
defect that kept the incubation clock at zero (see ``engine.search.research``). Instead a
track stamps the params *as they were the night it opened* and keeps measuring those,
whatever the population row says later. One open track per family at a time; when its
window fills, a verdict is recorded and the track closes.

A pass judges first and stamps second, so the family is re-opened on tonight's params in
the same pass that closed the old window and is never idle between windows. A family
therefore accumulates one verdict per 60 forward bars, each about a different draw of the
search, on bars that draw had never seen.

A family whose strategy needs intrabar bars is skipped rather than measured on ordinary
ones: ``wemof_tick``'s whole thesis is its intrabar pip exits, and computing its weights on
bar closes would produce a confident number for a strategy that does not exist.

**Storage is JSON Lines, one record per line, deliberately.** ``state/`` is committed on
every nightly run, and a SQLite file rewrites pages unpredictably, so each commit stores a
fresh ~250KB blob — around 140MB of git history a year for a file whose content barely
changes. One line per family means an untouched track is an untouched line and git stores
a delta instead. ``verdicts.jsonl`` is append-only for the same reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from engine.backtest.metrics import max_drawdown
from engine.core.config import Settings
from engine.data.store import OHLCVStore
from engine.monitor.decay import sharpe_lower_band
from engine.orchestration.population import Population
from engine.orchestration.promotion import (
    PromotionConfig,
    forward_returns,
    incubation_passed,
)
from engine.strategies import get_strategy
from engine.validation.stats import return_moments

# Verdicts are append-only, so without a cap the file grows forever inside a git-tracked
# state directory. Twenty per family keeps a few years of history at a few hundred KB.
MAX_VERDICTS_PER_FAMILY = 20

TRACKS_FILE = "tracks.jsonl"
VERDICTS_FILE = "verdicts.jsonl"

TRACK_FIELDS = (
    "family",
    "strategy",
    "symbol",
    "source",
    "timeframe",
    "params",
    "params_hash",
    "gate_metrics",
    "start_ts",
    "created_at",
)


def params_hash(params: dict) -> str:
    """Stable identity for a parameter set, so a re-opened track can be told apart."""
    return sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A half-written final line (killed mid-append) must not take the step down.
            continue
    return out


@dataclass(slots=True)
class ShadowSummary:
    n_opened: int = 0
    n_tracking: int = 0
    n_closed: int = 0
    n_would_have_passed: int = 0
    n_skipped_intrabar: int = 0
    n_errors: int = 0
    errors: list[str] = field(default_factory=list)


class ShadowLedger:
    """Open tracks and closed verdicts under ``state/shadow/``.

    Tracks are held in memory and written by :meth:`save`; verdicts are appended as they
    are recorded, so a crash mid-pass loses at most the open-track bookkeeping, never a
    verdict that was already measured.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tracks: dict[str, dict] = {
            row["family"]: row for row in _read_jsonl(self.root / TRACKS_FILE)
        }

    # -- open tracks -------------------------------------------------------------------

    def open_track(self, record: dict, start_ts: int) -> None:
        """Stamp a rejected candidate's current params and start its forward window."""
        self.tracks[record["family"]] = {
            "family": record["family"],
            "strategy": record["strategy"],
            "symbol": record["symbol"],
            "source": record["source"],
            "timeframe": record["timeframe"],
            "params": record["params"],
            "params_hash": params_hash(record["params"]),
            "gate_metrics": record["metrics"],
            "start_ts": int(start_ts),
            "created_at": datetime.now(UTC).isoformat(),
        }

    def has_track(self, family: str) -> bool:
        return family in self.tracks

    def close_track(self, family: str) -> None:
        self.tracks.pop(family, None)

    def save(self) -> None:
        """Rewrite ``tracks.jsonl``, one family per line, sorted so diffs stay stable."""
        lines = [
            json.dumps({k: track[k] for k in TRACK_FIELDS}, sort_keys=True)
            for _, track in sorted(self.tracks.items())
        ]
        path = self.root / TRACKS_FILE
        path.write_text("\n".join(lines) + ("\n" if lines else ""))

    # -- verdicts ----------------------------------------------------------------------

    def append_verdict(self, **fields) -> None:
        fields["closed_at"] = datetime.now(UTC).isoformat()
        path = self.root / VERDICTS_FILE
        with path.open("a") as handle:
            handle.write(json.dumps(fields, sort_keys=True) + "\n")
        self._prune(fields["family"])

    def _prune(self, family: str) -> None:
        """Drop the oldest verdicts for ``family`` past the cap (rarely triggers)."""
        rows = _read_jsonl(self.root / VERDICTS_FILE)
        kept_for_family = [i for i, r in enumerate(rows) if r.get("family") == family]
        if len(kept_for_family) <= MAX_VERDICTS_PER_FAMILY:
            return
        drop = set(kept_for_family[: len(kept_for_family) - MAX_VERDICTS_PER_FAMILY])
        remaining = [r for i, r in enumerate(rows) if i not in drop]
        (self.root / VERDICTS_FILE).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in remaining)
        )

    def verdicts(self) -> list[dict]:
        """Every recorded verdict, newest first."""
        return list(reversed(_read_jsonl(self.root / VERDICTS_FILE)))

    def tally(self) -> tuple[int, int]:
        """``(n_verdicts, n_would_have_passed)`` over everything recorded."""
        rows = _read_jsonl(self.root / VERDICTS_FILE)
        return len(rows), sum(1 for r in rows if r.get("would_have_passed"))


def _requires_intrabar(strategy: str) -> bool:
    try:
        return bool(getattr(get_strategy(strategy), "requires_intrabar", False))
    except Exception:
        return False


def _rejected(record: dict) -> bool:
    """A population row the gate turned down, as opposed to one it has not judged."""
    return record["status"] == "candidate" and not (record.get("metrics") or {}).get("passed")


def shadow_step(settings: Settings, *, cfg: PromotionConfig | None = None) -> ShadowSummary:
    """One pass: judge the tracks whose window has filled, then stamp the new rejects."""
    cfg = cfg or PromotionConfig()
    store = OHLCVStore(settings.data_dir / "ohlcv")
    population = Population(settings.state_dir / "population.db")
    ledger = ShadowLedger(settings.state_dir / "shadow")
    summary = ShadowSummary()
    try:
        # 1. Close any track whose forward window is now long enough to judge.
        for track in list(ledger.tracks.values()):
            record = {
                "family": track["family"],
                "strategy": track["strategy"],
                "symbol": track["symbol"],
                "source": track["source"],
                "timeframe": track["timeframe"],
                "params": track["params"],
                "metrics": track["gate_metrics"],
            }
            try:
                fwd = forward_returns(store, record, cfg, track["start_ts"])
            except Exception as exc:  # a param set the strategy no longer accepts
                summary.n_errors += 1
                summary.errors.append(f"{track['family']}: {type(exc).__name__}: {exc}")
                ledger.close_track(track["family"])
                continue
            if fwd is None or len(fwd) < cfg.min_incubation_bars:
                summary.n_tracking += 1
                continue

            gate = track["gate_metrics"]
            moments = return_moments(fwd)
            passed = incubation_passed(fwd, gate, cfg)
            ppy = float(gate.get("ppy", 365.0))
            equity = pd.Series(np.cumprod(1.0 + np.asarray(fwd, dtype=float)))
            ledger.append_verdict(
                family=track["family"],
                strategy=track["strategy"],
                symbol=track["symbol"],
                source=track["source"],
                timeframe=track["timeframe"],
                params=track["params"],
                params_hash=track["params_hash"],
                start_ts=track["start_ts"],
                n_bars=len(fwd),
                fwd_sharpe_per_bar=float(moments.sharpe),
                fwd_sharpe_annual=float(moments.sharpe * np.sqrt(ppy)),
                fwd_cum_return=float(equity.iloc[-1] - 1.0),
                fwd_max_drawdown=float(max_drawdown(equity)),
                would_have_passed=bool(passed),
                required_lower_band=_lower_band(gate, cfg),
                gate_dsr=_float_or_none(gate.get("dsr")),
                gate_pbo=_float_or_none(gate.get("pbo")),
                gate_spa=_float_or_none(gate.get("spa_p")),
                bt_sharpe_annual=_float_or_none(gate.get("sharpe_annual")),
            )
            ledger.close_track(track["family"])
            summary.n_closed += 1
            summary.n_would_have_passed += int(passed)

        # 2. Open a track for every reject that has none, stamping tonight's params.
        for record in population.list("candidate"):
            if not _rejected(record) or ledger.has_track(record["family"]):
                continue
            if _requires_intrabar(record["strategy"]):
                # Its edge lives in intrabar exits no bar-close recomputation can see.
                summary.n_skipped_intrabar += 1
                continue
            last_ts = store.last_ts(record["source"], record["symbol"], record["timeframe"])
            if last_ts is None:
                continue
            ledger.open_track(record, last_ts)
            summary.n_opened += 1
            summary.n_tracking += 1

        ledger.save()
    finally:
        population.close()
    return summary


def _lower_band(gate: dict, cfg: PromotionConfig) -> float | None:
    """The same bar a promoted strategy's forward Sharpe must clear, for comparison."""
    try:
        return float(
            sharpe_lower_band(
                gate.get("sharpe_per_bar", 0.0),
                int(gate.get("n", 0)),
                gate.get("skew", 0.0),
                gate.get("kurtosis", 3.0),
                z=cfg.band_z,
            )
        )
    except Exception:
        return None


def _float_or_none(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "MAX_VERDICTS_PER_FAMILY",
    "ShadowLedger",
    "ShadowSummary",
    "params_hash",
    "shadow_step",
]
