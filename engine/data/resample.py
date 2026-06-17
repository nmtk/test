"""Resample OHLCV bars to a higher timeframe.

Used to derive 4h/1d (and similar) from 1h on venues where the native higher-timeframe
candle endpoint is unreliable — notably bitbank, whose ccxt 4h/1d requests return error
10000. Deriving from a single base timeframe also guarantees the timeframes are mutually
consistent. Buckets are epoch-anchored (matching :func:`engine.core.clock.floor_to_timeframe`)
and only *complete* buckets are emitted, so no partial/forming bar ever leaks in.
"""

from __future__ import annotations

import pandas as pd

from engine.core.clock import timeframe_seconds
from engine.data.schema import OHLCV_COLUMNS, empty_ohlcv, normalize_ohlcv


def resample_ohlcv(df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
    """Aggregate ``source_tf`` bars into ``target_tf`` bars (target must be a multiple)."""
    df = normalize_ohlcv(df)
    src = timeframe_seconds(source_tf)
    tgt = timeframe_seconds(target_tf)
    if tgt % src != 0:
        raise ValueError(f"target timeframe {target_tf} is not a multiple of {source_tf}")
    ratio = tgt // src
    if ratio == 1 or len(df) == 0:
        return df

    indexed = df.copy()
    indexed.index = pd.to_datetime(indexed["ts"], unit="ms", utc=True)
    agg = indexed.resample(f"{tgt}s", label="left", closed="left", origin="epoch").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        count=("close", "size"),
    )
    agg = agg[agg["count"] == ratio]  # complete buckets only — never emit a forming bar
    if len(agg) == 0:
        return empty_ohlcv()
    # Epoch-ms, robust to the index's datetime resolution (ns vs ms differs across pandas).
    epoch = pd.Timestamp(0, tz="UTC")
    agg["ts"] = ((agg.index - epoch) // pd.Timedelta(milliseconds=1)).astype("int64")
    agg = agg.reset_index(drop=True)  # drop the datetime index (named "ts") to avoid clash
    return normalize_ohlcv(agg[OHLCV_COLUMNS])
