"""Canonical OHLCV schema and normalization helpers.

In-memory and on-disk, a bar series is a :class:`pandas.DataFrame` with columns:

    ts      int64   bar open time, epoch **milliseconds**, UTC
    open    float64
    high    float64
    low     float64
    close   float64
    volume  float64

Integer-ms timestamps are portable, exact, and make idempotent dedupe trivial. Use
:func:`with_datetime` when a tz-aware index is needed (plots, calendars).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

OHLCV_COLUMNS: list[str] = ["ts", "open", "high", "low", "close", "volume"]
_FLOAT_COLUMNS = ["open", "high", "low", "close", "volume"]


def empty_ohlcv() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in _FLOAT_COLUMNS})
    df.insert(0, "ts", pd.Series(dtype="int64"))
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame to the canonical schema.

    Ensures column presence/order, int64-ms ``ts``, float prices, drops rows with NaN
    timestamps, sorts ascending, and de-duplicates on ``ts`` keeping the last value.
    """
    if df is None or len(df) == 0:
        return empty_ohlcv()

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")

    out = df[OHLCV_COLUMNS].copy()
    out = out.dropna(subset=["ts"])
    out["ts"] = out["ts"].astype("int64")
    for col in _FLOAT_COLUMNS:
        out[col] = out[col].astype("float64")
    out = out.sort_values("ts", kind="mergesort")
    out = out.drop_duplicates(subset="ts", keep="last")
    return out.reset_index(drop=True)


def merge_ohlcv(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Idempotent upsert: union two frames, dedupe on ``ts`` (incoming wins)."""
    existing = normalize_ohlcv(existing)
    incoming = normalize_ohlcv(incoming)
    if len(existing) == 0:
        return incoming
    if len(incoming) == 0:
        return existing
    combined = pd.concat([existing, incoming], ignore_index=True)
    # incoming appended last, so keep="last" lets fresh bars overwrite stale ones.
    return normalize_ohlcv(combined)


def with_datetime(df: pd.DataFrame, column: str = "dt") -> pd.DataFrame:
    """Return a copy with an added tz-aware (UTC) datetime column derived from ``ts``."""
    out = df.copy()
    out[column] = pd.to_datetime(out["ts"], unit="ms", utc=True)
    return out


def to_epoch_ms(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return int(ts.timestamp() * 1000)
