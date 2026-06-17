"""Feature terminals available to GP programs, built causally from OHLCV bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: tuple[str, ...] = ("close", "ret", "logret", "hl_range", "volume_z")


def build_features(bars: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the terminal feature series a program may reference (all causal)."""
    close = bars["close"].astype(float).reset_index(drop=True)
    high = bars["high"].astype(float).reset_index(drop=True)
    low = bars["low"].astype(float).reset_index(drop=True)
    volume = bars["volume"].astype(float).reset_index(drop=True)

    ret = close.pct_change().fillna(0.0)
    logret = np.log(close).diff().fillna(0.0)
    hl_range = ((high - low) / close.replace(0, np.nan)).fillna(0.0)
    vol_mean = volume.rolling(50, min_periods=10).mean()
    vol_std = volume.rolling(50, min_periods=10).std(ddof=0)
    volume_z = ((volume - vol_mean) / vol_std.replace(0, np.nan)).fillna(0.0)

    return {
        "close": close,
        "ret": ret,
        "logret": logret,
        "hl_range": hl_range,
        "volume_z": volume_z,
    }
