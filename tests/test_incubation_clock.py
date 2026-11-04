"""The incubation clock must survive the nightly research run.

This is the regression suite for the defect that kept ``state/champions.json`` empty for
five months of nightly runs. ``_record_for`` only preserved ``incubation_start_ts`` for a
*champion* whose params were unchanged. An **incubating** row fell through to the final
branch and had its clock reset to the newest bar — every night, by the very run that was
supposed to be leaving it alone to accumulate forward evidence.

The forward window is ``bars with ts > incubation_start_ts``. Resetting the clock to the
last bar each night pins that window at zero or one bar, so
``PromotionConfig.min_incubation_bars`` (60) is unreachable *in principle*: not a matter of
waiting longer, a matter of the counter restarting before it can count. The observed
population snapshots show exactly that signature — the clock stepping 2026-10-26 ->
10-28 -> 10-31 -> 11-02 while the forward window never left one bar, repeated gate passes,
and never one promotion.

Two invariants pin it down:
  1. research does not overwrite an incumbent's row (clock, params or reference metrics);
  2. incubation nevertheless terminates — a full window that fails demotes the row, so a
     held family cannot be frozen for good.
"""

from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pandas as pd
import pytest

from engine.core.config import load_settings
from engine.data.store import OHLCVStore
from engine.orchestration.population import Population, StrategyRecord
from engine.orchestration.promotion import PromotionConfig, promote_step
from engine.search.research import _record_for, held_incumbent

DAY_MS = 86_400_000


def _report(passed: bool = True, params: dict | None = None):
    """A stand-in for a ValidationReport: only the fields _record_for reads."""
    return types.SimpleNamespace(
        passed=passed,
        best_params=params if params is not None else {"window": 20},
        dsr=0.99,
        pbo=0.05,
        spa_p=0.01,
        cpcv_median=1.5,
        sharpe_annual=2.0,
        sharpe_per_bar=0.1,
        mean_per_bar=0.001,
        std_per_bar=0.01,
        skew=0.0,
        kurtosis=3.0,
        max_drawdown=-0.05,
        n_bars=1200,
    )


def _kwargs(existing, last_ts):
    return dict(
        family="tsmom:BTC/JPY:1d",
        strat_name="tsmom",
        symbol="BTC/JPY",
        source="bitbank",
        tf="1d",
        ppy=365.0,
        last_ts=last_ts,
        existing=existing,
    )


def _incumbent(status="incubating", start_ts=1_000 * DAY_MS, params=None, research_only=False):
    return {
        "family": "tsmom:BTC/JPY:1d",
        "strategy": "tsmom",
        "symbol": "BTC/JPY",
        "source": "bitbank",
        "timeframe": "1d",
        "params": params if params is not None else {"window": 20},
        "status": status,
        "metrics": {"sharpe_per_bar": 0.1, "n": 1200, "research_only": research_only},
        "incubation_start_ts": start_ts,
        "champion_start_ts": None,
    }


# -- the clock ------------------------------------------------------------------------


def test_nightly_rediscovery_does_not_restart_the_clock():
    """The exact defect: night two must not move a clock set on night one."""
    start = 1_000 * DAY_MS
    row = _incumbent(start_ts=start)
    # Night two finds a *different* best configuration, as a stochastic search will.
    rec = _record_for(_report(params={"window": 55}), **_kwargs(row, last_ts=1_001 * DAY_MS))
    assert rec.incubation_start_ts == start, "the forward window would restart at zero"
    assert rec.status == "incubating"
    assert rec.params == {"window": 20}, "the row under test keeps the params being tested"


def test_the_window_actually_grows_across_consecutive_nights():
    """Sixty nights of re-discovery must leave sixty bars of forward evidence, not one."""
    start = 1_000 * DAY_MS
    row = _incumbent(start_ts=start)
    for night in range(1, 61):
        rec = _record_for(_report(), **_kwargs(row, last_ts=(1_000 + night) * DAY_MS))
        row = {**row, **dataclasses.asdict(rec)}
        row["timeframe"] = rec.timeframe
    last_ts = (1_000 + 60) * DAY_MS
    forward_bars = (last_ts - row["incubation_start_ts"]) // DAY_MS
    assert forward_bars == 60, f"only {forward_bars} forward bars accumulated over 60 nights"


def test_reference_metrics_are_not_refit_nightly():
    """The incubation band is compared against the metrics the gate validated, not tonight's."""
    row = _incumbent()
    row["metrics"] = {"sharpe_per_bar": 0.42, "n": 999, "research_only": False}
    rec = _record_for(_report(), **_kwargs(row, last_ts=1_001 * DAY_MS))
    assert rec.metrics["sharpe_per_bar"] == 0.42 and rec.metrics["n"] == 999


def test_a_champion_keeps_its_clocks_even_when_the_search_disagrees():
    row = _incumbent(status="champion", start_ts=900 * DAY_MS)
    row["champion_start_ts"] = 950 * DAY_MS
    rec = _record_for(_report(params={"window": 7}), **_kwargs(row, last_ts=1_001 * DAY_MS))
    assert rec.status == "champion"
    assert (rec.incubation_start_ts, rec.champion_start_ts) == (900 * DAY_MS, 950 * DAY_MS)


def test_a_free_family_still_starts_its_clock_at_the_last_bar():
    for existing in (None, _incumbent(status="candidate"), _incumbent(status="retired")):
        rec = _record_for(_report(), **_kwargs(existing, last_ts=1_001 * DAY_MS))
        assert rec.status == "incubating"
        assert rec.incubation_start_ts == 1_001 * DAY_MS


def test_a_failed_gate_still_writes_a_candidate():
    rec = _record_for(_report(passed=False), **_kwargs(None, last_ts=1_001 * DAY_MS))
    assert rec.status == "candidate" and rec.incubation_start_ts is None


# -- what counts as held --------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [("incubating", True), ("champion", True), ("candidate", False), ("retired", False)],
)
def test_held_statuses(status, expected):
    assert held_incumbent(_incumbent(status=status)) is expected


def test_nothing_is_held_when_there_is_no_row():
    assert held_incumbent(None) is False
    assert held_incumbent({}) is False


def test_a_research_only_row_is_never_held():
    """The FSM deliberately never touches these, so holding one would freeze it for good.

    A research-only family (fred:DEXJPUS, wemof_tick) can pass the gate but can never be
    promoted — no venue can fill it. promote_step skips it before the forward test, which
    means nothing would ever hand the family back to research.
    """
    row = _incumbent(research_only=True)
    assert held_incumbent(row) is False
    rec = _record_for(_report(params={"window": 55}), **_kwargs(row, last_ts=1_001 * DAY_MS))
    assert rec.params == {"window": 55}, "research must stay free to keep searching it"


# -- the verdict ----------------------------------------------------------------------


def _settings_at(tmp_path):
    base = load_settings()
    return dataclasses.replace(base, state_dir=tmp_path / "state", data_dir=tmp_path / "data")


def _write_bars(
    tmp_path, n: int, *, trend: float = 0.0, noise: float = 0.0, whipsaw: bool = False
) -> int:
    """A deterministic series: a compounding trend momentum rides, or one it loses on.

    ``whipsaw`` steps between two levels every three bars, which a 5-bar momentum signal
    reads out of phase: the forward Sharpe is about -0.7, so incubation genuinely fails.

    ``noise`` is needed wherever a test asserts something about a confidence *band*. A
    perfectly smooth series has zero return variance, so ``mean / std`` blows up to a
    Sharpe of ~1e16 and clears any band for a reason that has nothing to do with edge.
    """
    index = np.arange(n)
    if whipsaw:
        close = 1_000_000.0 * (1.0 + 0.02 * ((index // 3) % 2))
    else:
        steps = np.full(n, 1.0 + trend)
        if noise:
            steps = steps + np.random.default_rng(0).normal(0.0, noise, n)
        close = 1_000_000.0 * np.cumprod(steps)
    frame = pd.DataFrame(
        {
            "ts": index.astype("int64") * DAY_MS + 1_700_000_000_000,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )
    store = OHLCVStore(tmp_path / "data" / "ohlcv")
    path = store.path("bitbank", "BTC/JPY", "1d")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    assert len(store.read("bitbank", "BTC/JPY", "1d")) == n
    return int(frame["ts"].iloc[0])


def _seed_incubating(settings, start_ts: int, *, sharpe_per_bar: float):
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    population = Population(settings.state_dir / "population.db")
    population.upsert(
        StrategyRecord(
            family="tsmom:BTC/JPY:1d",
            strategy="tsmom",
            symbol="BTC/JPY",
            source="bitbank",
            timeframe="1d",
            # A monotone series makes tsmom always long, so forward returns track it.
            params={"lookback": 5},
            status="incubating",
            metrics={
                "ppy": 365.0,
                "n": 1200,
                "sharpe_per_bar": sharpe_per_bar,
                "skew": 0.0,
                "kurtosis": 3.0,
                "max_drawdown": -0.05,
                "fee_bps": 0.0,
                "slippage_bps": 0.0,
                "research_only": False,
            },
            incubation_start_ts=start_ts,
        )
    )
    population.close()


def _status(settings) -> dict:
    population = Population(settings.state_dir / "population.db")
    try:
        return population.get("tsmom:BTC/JPY:1d")
    finally:
        population.close()


def test_a_short_window_is_left_alone_to_keep_incubating(tmp_path):
    """Fewer forward bars than required is "not yet", not a failure."""
    settings = _settings_at(tmp_path)
    first_ts = _write_bars(tmp_path, 400, whipsaw=True)  # losing, but only 9 bars forward
    _seed_incubating(settings, first_ts + 390 * DAY_MS, sharpe_per_bar=0.5)
    summary = promote_step(settings, cfg=PromotionConfig(min_incubation_bars=60))
    assert (summary.n_promoted, summary.n_rejected) == (0, 0)
    assert _status(settings)["status"] == "incubating"


def test_a_full_window_that_fails_is_demoted_so_the_family_reopens(tmp_path):
    """Otherwise research would never search this family again (it holds off on incumbents)."""
    settings = _settings_at(tmp_path)
    first_ts = _write_bars(tmp_path, 400, whipsaw=True)  # forward Sharpe is about -0.7
    _seed_incubating(settings, first_ts + 300 * DAY_MS, sharpe_per_bar=0.5)
    summary = promote_step(settings, cfg=PromotionConfig(min_incubation_bars=60))
    assert summary.n_rejected == 1 and summary.n_promoted == 0
    row = _status(settings)
    assert row["status"] == "candidate", "the slot must reopen for a fresh search"
    assert row["incubation_start_ts"] is None
    assert held_incumbent(row) is False


def test_a_full_window_that_holds_up_is_promoted(tmp_path):
    """The other half of the verdict: the path that was unreachable before the clock fix."""
    settings = _settings_at(tmp_path)
    first_ts = _write_bars(tmp_path, 400, trend=0.01, noise=0.01)
    _seed_incubating(settings, first_ts + 300 * DAY_MS, sharpe_per_bar=0.05)
    summary = promote_step(settings, cfg=PromotionConfig(min_incubation_bars=60))
    assert summary.n_promoted == 1 and summary.n_rejected == 0
    row = _status(settings)
    assert row["status"] == "champion" and row["champion_start_ts"] is not None
    champions = json.loads((settings.state_dir / "champions.json").read_text())
    assert champions["allocations"], "a promoted strategy must receive an allocation"
