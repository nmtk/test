"""Intrabar bid/ask path: what happened *inside* each bar, kept from the same ticks.

The 5m OHLCV series is built from the mid and so throws away exactly the two things a
scalp's exit depends on:

* **order** — whether a 1-pip take-profit or a 5-pip stop was touched first inside a bar;
* **sides** — a long exits on the bid and a short on the ask, so the spread is part of
  every fill, and the mid hides it.

This module keeps a compact one-minute path of *both* sides, built from the Dukascopy
ticks that already feed the 5m bars. Per minute: bid and ask open/high/low/close, plus

* ``bid_low_step`` — the largest single step by which a bid tick undercut the lowest bid
  seen earlier in the same minute;
* ``ask_high_step`` — the same for the ask making new highs.

What one-minute resolution does and does not decide (the simulator in
``engine/backtest/intrabar.py`` relies on each of these):

* Whether a price level is touched in a minute is **exact** — iff the minute's high/low
  reaches it.
* When a take-profit *and* a stop are both touched in one minute, their order is unknown.
  The simulator resolves that against the trade (stop first).
* A sell stop fills at the first bid at or below it, which can be worse than the level.
  Every earlier tick of the minute was above the level, so that fill is a new low for
  the minute and lies at most ``bid_low_step`` below the level. Filling there is never
  better than the tick-exact fill. It is the minute's *largest* such step, though, not
  the one at the crossing, so it costs edge: against a tick replay of synthetic streams
  about a quarter of a pip per stop exit (take-profit, time and gap exits are exact).
  Storing every record low per minute would make stops exact too, at a variable-length
  column's complexity; it is not worth it while stops are the minority of exits.

Why not keep the raw ticks: three years of USD/JPY is on the order of 1.5GB of tick
records, and this repo commits its data directory. Stored as integer points with every
column but one relative to its neighbour (spreads and ranges are small integers that
compress well), a month measured ~0.4MB on synthetic ticks — the same order as the 5m
series it sits next to.

Layout, one file per month like every sub-hourly series::

    data/intrabar/{source}/{symbol}/{step}/{YYYY-MM}.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from engine.core.clock import timeframe_seconds
from engine.data.dukascopy import point_divisor
from engine.data.store import OHLCVStore, month_key, sanitize_symbol

STEP = "1m"
STEP_MS = timeframe_seconds(STEP) * 1000

# Sources whose feed carries both sides of every quote. A mid-only feed cannot build this.
INTRABAR_SOURCES: frozenset[str] = frozenset({"dukascopy"})

# Label used in sync output so the path is never mistaken for an OHLCV timeframe.
PATH_LABEL = f"{STEP}-path"

# Same history depth the OHLCV sync seeds a brand-new series with.
_SEED_DAYS = 365 * 3

# Stored columns. ``bid_open`` is absolute integer points; every other price column is
# relative, so the file holds small integers rather than six-digit ones.
_COLUMNS: tuple[str, ...] = (
    "ts",
    "bid_open",
    "bid_high_d",  # bid_high - bid_open           (>= 0)
    "bid_low_d",  # bid_open - bid_low             (>= 0)
    "bid_close_d",  # bid_close - bid_open
    "ask_open_s",  # ask_open - bid_open           (spread at the open)
    "ask_high_s",  # ask_high - bid_high
    "ask_low_s",  # ask_low - bid_low
    "ask_close_s",  # ask_close - bid_close        (spread at the close)
    "bid_low_step",
    "ask_high_step",
)


def pip_size(symbol: str) -> float:
    """One pip in price units: 0.01 for JPY quotes, 0.0001 otherwise (10 feed points)."""
    return 10.0 / point_divisor(symbol)


def empty_path_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="int64" if c == "ts" else "int32") for c in _COLUMNS})


def ticks_to_path(ticks: pd.DataFrame, symbol: str, step: str = STEP) -> pd.DataFrame:
    """Aggregate ``ts``/``bid``/``ask`` ticks into the compact per-``step`` path frame."""
    if len(ticks) == 0:
        return empty_path_frame()
    divisor = point_divisor(symbol)
    step_ms = timeframe_seconds(step) * 1000
    ticks = ticks.sort_values("ts", kind="mergesort")
    ts = ticks["ts"].to_numpy(dtype="int64")
    bid = np.rint(ticks["bid"].to_numpy(dtype=float) * divisor).astype("int64")
    ask = np.rint(ticks["ask"].to_numpy(dtype=float) * divisor).astype("int64")
    bucket = (ts // step_ms) * step_ms

    frame = pd.DataFrame({"bucket": bucket, "bid": bid, "ask": ask})
    g = frame.groupby("bucket", sort=True)
    # Extremes seen *before* each tick within its own minute. A minute's first tick has
    # none: how far it lands from the previous minute is the open gap, which the
    # simulator fills at the open, exactly.
    prior_low = g["bid"].cummin().groupby(frame["bucket"]).shift(1)
    prior_high = g["ask"].cummax().groupby(frame["bucket"]).shift(1)
    frame["low_step"] = (prior_low - frame["bid"]).clip(lower=0).fillna(0)
    frame["high_step"] = (frame["ask"] - prior_high).clip(lower=0).fillna(0)
    g = frame.groupby("bucket", sort=True)
    bid_o, bid_h, bid_l, bid_c = (g["bid"].first(), g["bid"].max(), g["bid"].min(), g["bid"].last())
    ask_o, ask_h, ask_l, ask_c = (g["ask"].first(), g["ask"].max(), g["ask"].min(), g["ask"].last())
    out = pd.DataFrame(
        {
            "ts": bid_o.index.to_numpy(dtype="int64"),
            "bid_open": bid_o.to_numpy(),
            "bid_high_d": (bid_h - bid_o).to_numpy(),
            "bid_low_d": (bid_o - bid_l).to_numpy(),
            "bid_close_d": (bid_c - bid_o).to_numpy(),
            "ask_open_s": (ask_o - bid_o).to_numpy(),
            "ask_high_s": (ask_h - bid_h).to_numpy(),
            "ask_low_s": (ask_l - bid_l).to_numpy(),
            "ask_close_s": (ask_c - bid_c).to_numpy(),
            "bid_low_step": g["low_step"].max().to_numpy(),
            "ask_high_step": g["high_step"].max().to_numpy(),
        }
    )
    return _normalize(out)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[list(_COLUMNS)].copy()
    frame["ts"] = frame["ts"].astype("int64")
    for col in _COLUMNS[1:]:
        frame[col] = frame[col].astype("int32")
    frame = frame.drop_duplicates("ts", keep="last").sort_values("ts", kind="mergesort")
    return frame.reset_index(drop=True)


def _merge(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Upsert keyed on ``ts``: the incoming row wins (it may be a re-fetched final minute)."""
    if len(existing) == 0:
        return _normalize(incoming)
    return _normalize(pd.concat([existing, incoming], ignore_index=True))


# -- the float view the simulator reads ----------------------------------------------------


def _frozen(values: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(values, dtype=float)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, slots=True)
class IntrabarPath:
    """Decoded path in price units, one row per ``step_ms`` sub-bar, ``ts`` increasing.

    Immutable (read-only arrays) so it can ride along in ``DataFrame.attrs``: pandas
    deep-copies ``attrs`` on nearly every operation, and ``__deepcopy__`` returning
    ``self`` keeps that from copying a million-row path on each column access.
    """

    ts: np.ndarray
    step_ms: int
    pip: float
    bid_open: np.ndarray
    bid_high: np.ndarray
    bid_low: np.ndarray
    bid_close: np.ndarray
    ask_open: np.ndarray
    ask_high: np.ndarray
    ask_low: np.ndarray
    ask_close: np.ndarray
    bid_low_step: np.ndarray
    ask_high_step: np.ndarray

    def __deepcopy__(self, memo) -> IntrabarPath:
        return self

    def __len__(self) -> int:
        return len(self.ts)

    @property
    def end_ts(self) -> int:
        """End (exclusive) of the last sub-bar, or 0 for an empty path."""
        return int(self.ts[-1]) + self.step_ms if len(self.ts) else 0

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, symbol: str, step_ms: int = STEP_MS) -> IntrabarPath:
        frame = _normalize(frame)
        scale = 1.0 / point_divisor(symbol)
        bid_open = frame["bid_open"].to_numpy(dtype="int64")
        bid_high = bid_open + frame["bid_high_d"].to_numpy(dtype="int64")
        bid_low = bid_open - frame["bid_low_d"].to_numpy(dtype="int64")
        bid_close = bid_open + frame["bid_close_d"].to_numpy(dtype="int64")
        ts = frame["ts"].to_numpy(dtype="int64").copy()
        ts.setflags(write=False)
        return cls(
            ts=ts,
            step_ms=int(step_ms),
            pip=pip_size(symbol),
            bid_open=_frozen(bid_open * scale),
            bid_high=_frozen(bid_high * scale),
            bid_low=_frozen(bid_low * scale),
            bid_close=_frozen(bid_close * scale),
            ask_open=_frozen((bid_open + frame["ask_open_s"].to_numpy(dtype="int64")) * scale),
            ask_high=_frozen((bid_high + frame["ask_high_s"].to_numpy(dtype="int64")) * scale),
            ask_low=_frozen((bid_low + frame["ask_low_s"].to_numpy(dtype="int64")) * scale),
            ask_close=_frozen((bid_close + frame["ask_close_s"].to_numpy(dtype="int64")) * scale),
            bid_low_step=_frozen(frame["bid_low_step"].to_numpy(dtype="int64") * scale),
            ask_high_step=_frozen(frame["ask_high_step"].to_numpy(dtype="int64") * scale),
        )


# -- storage --------------------------------------------------------------------------------


class IntrabarStore:
    """Month-partitioned parquet store for path frames. Upserts are idempotent on ``ts``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def partition_dir(self, source: str, symbol: str, step: str = STEP) -> Path:
        return self.root / source / sanitize_symbol(symbol) / step

    def files(self, source: str, symbol: str, step: str = STEP) -> list[Path]:
        return sorted(self.partition_dir(source, symbol, step).glob("*.parquet"))

    def read_frame(
        self,
        source: str,
        symbol: str,
        step: str = STEP,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        paths = self.files(source, symbol, step)
        if not paths:
            return empty_path_frame()
        frame = _normalize(pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True))
        if start_ms is not None:
            frame = frame[frame["ts"] >= start_ms]
        if end_ms is not None:
            frame = frame[frame["ts"] < end_ms]
        return frame.reset_index(drop=True)

    def read(self, source: str, symbol: str, step: str = STEP) -> IntrabarPath:
        frame = self.read_frame(source, symbol, step)
        return IntrabarPath.from_frame(frame, symbol, timeframe_seconds(step) * 1000)

    def last_ts(self, source: str, symbol: str, step: str = STEP) -> int | None:
        paths = self.files(source, symbol, step)
        for path in reversed(paths):
            ts = pd.read_parquet(path, columns=["ts"])["ts"]
            if len(ts):
                return int(ts.max())
        return None

    def upsert(self, source: str, symbol: str, frame: pd.DataFrame, step: str = STEP) -> int:
        """Merge ``frame`` into the stored path; only the months it touches are rewritten."""
        incoming = _normalize(frame) if len(frame) else empty_path_frame()
        directory = self.partition_dir(source, symbol, step)
        directory.mkdir(parents=True, exist_ok=True)
        if len(incoming):
            keys = incoming["ts"].map(month_key)
            for key, chunk in incoming.groupby(keys, sort=True):
                path = directory / f"{key}.parquet"
                existing = pd.read_parquet(path) if path.is_file() else empty_path_frame()
                _merge(existing, chunk).to_parquet(
                    path, engine="pyarrow", index=False, compression="zstd"
                )
        return sum(
            len(pd.read_parquet(p, columns=["ts"])) for p in self.files(source, symbol, step)
        )


# -- sync -----------------------------------------------------------------------------------


def _first_bar_ts(ohlcv: OHLCVStore, source: str, symbol: str, timeframes) -> int | None:
    """Earliest stored bar of the series' finest timeframe — where the path should start."""
    if not timeframes:
        return None
    finest = min(timeframes, key=timeframe_seconds)
    files = ohlcv.files(source, symbol, finest)
    if not files:
        return None
    ts = pd.read_parquet(files[0], columns=["ts"])["ts"]
    return int(ts.min()) if len(ts) else None


def sync_intrabar(settings, universe, *, store=None, ohlcv=None, client_for=None) -> list:
    """Extend the path of every instrument whose bar feed carries both quote sides.

    Walks forward from the last stored minute exactly like the OHLCV sync (same hourly
    request budget per run), and starts a brand-new path where that instrument's bars
    start, so the two cover the same history once the backfill has caught up.
    """
    from engine.data.sources import make_source
    from engine.data.sync import SyncResult

    store = store or IntrabarStore(settings.data_dir / "intrabar")
    ohlcv = ohlcv or OHLCVStore(settings.data_dir / "ohlcv")
    clients: dict[str, object] = {}

    def client(source: str):
        if client_for is not None:
            return client_for(source)
        if source not in clients:
            clients[source] = make_source(source, settings)
        return clients[source]

    results = []
    for entry in universe.entries:
        spec = entry.live_source
        if spec.source not in INTRABAR_SOURCES:
            continue
        try:
            since = store.last_ts(spec.source, spec.symbol)
            if since is None:
                since = _first_bar_ts(ohlcv, spec.source, spec.symbol, entry.timeframes)
            if since is None:
                seed = datetime.now(UTC) - timedelta(days=_SEED_DAYS)
                since = int(seed.timestamp() * 1000)
            ticks = client(spec.source).fetch_ticks(spec.symbol, since_ms=since)
            total = store.upsert(spec.source, spec.symbol, ticks_to_path(ticks, spec.symbol))
            results.append(SyncResult(spec.source, spec.symbol, PATH_LABEL, total, ok=True))
        except Exception as exc:  # noqa: BLE001 - one bad series must not abort the run
            results.append(
                SyncResult(spec.source, spec.symbol, PATH_LABEL, 0, ok=False, error=str(exc))
            )
    return results


# -- attaching the path to bars ---------------------------------------------------------------

_ATTR = "intrabar_path"
_ATTR_BAR_MS = "intrabar_bar_ms"

# The fraction of bars the path must cover before an intrabar strategy is researched. A
# backfill walks forward from the oldest month, so a partially synced path covers only the
# start of history; researching that every night would spend trial-ledger budget (raising
# the family's DSR bar forever) on a fraction of the data the gate will eventually judge.
MIN_COVERAGE = 0.95


def coverage(bar_ts: np.ndarray, bar_ms: int, path: IntrabarPath) -> float:
    """Fraction of bars that have at least one path sub-bar inside them."""
    bar_ts = np.asarray(bar_ts, dtype="int64")
    if len(bar_ts) == 0 or len(path) == 0:
        return 0.0
    lo = np.searchsorted(path.ts, bar_ts, side="left")
    hi = np.searchsorted(path.ts, bar_ts + bar_ms, side="left")
    return float(np.mean(hi > lo))


def attach_intrabar(bars: pd.DataFrame, path: IntrabarPath, bar_ms: int) -> pd.DataFrame:
    """Return the bars the path covers, carrying the path for an intrabar strategy."""
    if len(path) == 0:
        out = bars.iloc[0:0].copy()
    else:
        keep = (bars["ts"] >= path.ts[0]) & (bars["ts"] + bar_ms <= path.end_ts)
        out = bars[keep].reset_index(drop=True)
    out.attrs[_ATTR] = path
    out.attrs[_ATTR_BAR_MS] = int(bar_ms)
    return out


def intrabar_of(bars: pd.DataFrame) -> tuple[IntrabarPath, int] | None:
    path = bars.attrs.get(_ATTR)
    bar_ms = bars.attrs.get(_ATTR_BAR_MS)
    if path is None or bar_ms is None:
        return None
    return path, int(bar_ms)


def prepare_intrabar_bars(
    bars: pd.DataFrame,
    store: IntrabarStore,
    *,
    source: str,
    symbol: str,
    timeframe: str,
    min_coverage: float = MIN_COVERAGE,
) -> tuple[pd.DataFrame | None, float]:
    """Load, check and attach the path for ``bars``; ``None`` when coverage is too low."""
    if source not in INTRABAR_SOURCES or len(bars) == 0:
        return None, 0.0
    bar_ms = timeframe_seconds(timeframe) * 1000
    if bar_ms % STEP_MS:
        return None, 0.0
    path = store.read(source, symbol)
    cov = coverage(bars["ts"].to_numpy(dtype="int64"), bar_ms, path)
    if cov < min_coverage:
        return None, cov
    return attach_intrabar(bars, path, bar_ms), cov
