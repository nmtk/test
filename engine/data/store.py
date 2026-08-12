"""Parquet-backed OHLCV store.

Layout — one file per series for coarse timeframes::

    data/ohlcv/{source}/{symbol}/{timeframe}.parquet

and one file *per month* for sub-hourly ones::

    data/ohlcv/{source}/{symbol}/{timeframe}/{YYYY-MM}.parquet

where ``symbol`` is filesystem-sanitized (``/`` -> ``_``). Writes are idempotent upserts
keyed on ``ts``, so a data-sync run that is delayed, retried, or overlapping never
corrupts or duplicates history — it just re-converges. This is what lets Job A run on a
best-effort cron.

Why sub-hourly series are partitioned
-------------------------------------
This repo commits its data directory, so every sync writes to git history. A single
5-minute series is large — three years of USD/JPY is ~225,000 bars and ~9.9MB — and the
sync job rewrites the whole file to append a handful of bars, 48 times a day.

Git's delta compression handles that far better than one might guess (appending to a
parquet leaves the earlier data pages byte-identical, so the deltas are small): measured
at **12.3 MiB of pack growth per simulated day**, not the ~475MB a naive "one new blob per
commit" estimate would suggest. But 12 MiB/day is still ~4.5GB/year, which a free public
repo cannot carry indefinitely.

Partitioning by month bounds the rewrite to the current month's file — ~380KB for a 5m
series instead of 9.9MB — which cuts that by more than an order of magnitude. Coarse
timeframes stay single-file: 1h/4h/1d series are small and partitioning them would add
file-count churn for no benefit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.core.clock import timeframe_seconds
from engine.data.schema import empty_ohlcv, merge_ohlcv, normalize_ohlcv

# Timeframes strictly faster than this are stored one file per month. One hour is the
# natural cut: at 1h a year is ~8,760 bars (~400KB), at 5m it is ~105,000.
PARTITION_BELOW_SECONDS = 3600


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_").replace(" ", "")


def is_partitioned(timeframe: str) -> bool:
    """True when ``timeframe`` is stored as monthly partitions rather than one file."""
    try:
        return timeframe_seconds(timeframe) < PARTITION_BELOW_SECONDS
    except ValueError:
        # Non-bar labels (e.g. the "1mo" macro series) are coarse by definition.
        return False


def month_key(ts_ms: int) -> str:
    """Partition key (``"2024-03"``) for an epoch-ms timestamp."""
    return pd.Timestamp(int(ts_ms), unit="ms", tz="UTC").strftime("%Y-%m")


class OHLCVStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- paths -------------------------------------------------------------------------

    def _series_dir(self, source: str, symbol: str) -> Path:
        return self.root / source / sanitize_symbol(symbol)

    def path(self, source: str, symbol: str, timeframe: str) -> Path:
        """The single-file path for this series (also the legacy path when partitioned)."""
        return self._series_dir(source, symbol) / f"{timeframe}.parquet"

    def partition_dir(self, source: str, symbol: str, timeframe: str) -> Path:
        return self._series_dir(source, symbol) / timeframe

    def partition_path(self, source: str, symbol: str, timeframe: str, key: str) -> Path:
        return self.partition_dir(source, symbol, timeframe) / f"{key}.parquet"

    def files(self, source: str, symbol: str, timeframe: str) -> list[Path]:
        """Every file holding data for this series, oldest first.

        A legacy single file is included even for a partitioned timeframe, so a store
        written before partitioning stays readable until the next upsert migrates it.
        """
        found: list[Path] = []
        legacy = self.path(source, symbol, timeframe)
        if legacy.is_file():
            found.append(legacy)
        if is_partitioned(timeframe):
            found.extend(sorted(self.partition_dir(source, symbol, timeframe).glob("*.parquet")))
        return found

    def exists(self, source: str, symbol: str, timeframe: str) -> bool:
        return bool(self.files(source, symbol, timeframe))

    # -- read --------------------------------------------------------------------------

    def read(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        """Read a bar series, optionally bounded to ``[start_ms, end_ms)`` (epoch ms)."""
        paths = self.files(source, symbol, timeframe)
        if not paths:
            return empty_ohlcv()
        frames = [pd.read_parquet(p) for p in paths]
        df = normalize_ohlcv(pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0])
        if start_ms is not None:
            df = df[df["ts"] >= start_ms]
        if end_ms is not None:
            df = df[df["ts"] < end_ms]
        return df.reset_index(drop=True)

    def last_ts(self, source: str, symbol: str, timeframe: str) -> int | None:
        """Largest stored ``ts`` (epoch ms), or ``None`` if empty — for incremental sync.

        Reads only the newest partition when partitioned, so the hot path of an
        incremental sync never loads years of history just to find where it left off.
        """
        paths = self.files(source, symbol, timeframe)
        if not paths:
            return None
        newest = normalize_ohlcv(pd.read_parquet(paths[-1]))
        if len(newest) == 0:
            # An empty newest partition is degenerate; fall back to the whole series.
            df = self.read(source, symbol, timeframe)
            return int(df["ts"].iloc[-1]) if len(df) else None
        return int(newest["ts"].iloc[-1])

    # -- write -------------------------------------------------------------------------

    def upsert(self, source: str, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Merge ``df`` into the stored series. Returns the resulting row count."""
        if not is_partitioned(timeframe):
            path = self.path(source, symbol, timeframe)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged = merge_ohlcv(self.read(source, symbol, timeframe), df)
            merged.to_parquet(path, engine="pyarrow", index=False)
            return len(merged)
        return self._upsert_partitioned(source, symbol, timeframe, df)

    def _upsert_partitioned(
        self, source: str, symbol: str, timeframe: str, df: pd.DataFrame
    ) -> int:
        incoming = normalize_ohlcv(df)
        directory = self.partition_dir(source, symbol, timeframe)
        directory.mkdir(parents=True, exist_ok=True)

        # A series written before partitioning is folded in once, then removed, so the
        # data is never present in two places at the same time.
        legacy = self.path(source, symbol, timeframe)
        if legacy.is_file():
            incoming = merge_ohlcv(normalize_ohlcv(pd.read_parquet(legacy)), incoming)

        if len(incoming):
            keys = incoming["ts"].map(month_key)
            # Only months present in the incoming data are rewritten — that is the whole
            # point of partitioning, so an append touches one small file.
            for key, chunk in incoming.groupby(keys, sort=True):
                path = self.partition_path(source, symbol, timeframe, str(key))
                existing = (
                    normalize_ohlcv(pd.read_parquet(path)) if path.is_file() else empty_ohlcv()
                )
                merge_ohlcv(existing, chunk).to_parquet(path, engine="pyarrow", index=False)

        if legacy.is_file():
            legacy.unlink()

        total = 0
        for path in sorted(directory.glob("*.parquet")):
            total += len(pd.read_parquet(path, columns=["ts"]))
        return total
