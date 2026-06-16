"""Parquet-backed OHLCV store.

Layout::

    data/ohlcv/{source}/{symbol}/{timeframe}.parquet

where ``symbol`` is filesystem-sanitized (``/`` -> ``_``). Writes are idempotent upserts
keyed on ``ts``, so a data-sync run that is delayed, retried, or overlapping never
corrupts or duplicates history — it just re-converges. This is what lets Job A run on a
best-effort cron.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.data.schema import empty_ohlcv, merge_ohlcv, normalize_ohlcv


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_").replace(" ", "")


class OHLCVStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, source: str, symbol: str, timeframe: str) -> Path:
        return self.root / source / sanitize_symbol(symbol) / f"{timeframe}.parquet"

    def exists(self, source: str, symbol: str, timeframe: str) -> bool:
        return self.path(source, symbol, timeframe).exists()

    def read(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        """Read a bar series, optionally bounded to ``[start_ms, end_ms)`` (epoch ms)."""
        path = self.path(source, symbol, timeframe)
        if not path.exists():
            return empty_ohlcv()
        df = pd.read_parquet(path)
        df = normalize_ohlcv(df)
        if start_ms is not None:
            df = df[df["ts"] >= start_ms]
        if end_ms is not None:
            df = df[df["ts"] < end_ms]
        return df.reset_index(drop=True)

    def last_ts(self, source: str, symbol: str, timeframe: str) -> int | None:
        """Largest stored ``ts`` (epoch ms), or ``None`` if empty — for incremental sync."""
        df = self.read(source, symbol, timeframe)
        if len(df) == 0:
            return None
        return int(df["ts"].iloc[-1])

    def upsert(self, source: str, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Merge ``df`` into the stored series. Returns the resulting row count."""
        path = self.path(source, symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.read(source, symbol, timeframe)
        merged = merge_ohlcv(existing, df)
        merged.to_parquet(path, engine="pyarrow", index=False)
        return len(merged)
