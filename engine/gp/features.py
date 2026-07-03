"""Feature terminals available to GP programs, built causally from OHLCV bars.

Terminals come in two groups:

* **Base** — derived from the target's own OHLCV (returns, range, volume).
* **Context** — cross-asset ``ctx_*`` columns stamped onto the bars by
  :class:`engine.features.context.ContextAugmenter` (risk-on/off, dollar, rates, gold,
  BTC market factor). These let evolution discover *flow-conditional* alphas — e.g.
  "trend works only when the dollar is weak" — instead of price patterns alone.

When a ``ctx_*`` column is absent from ``bars`` (context not synced / not augmented) the
terminal evaluates to a neutral constant 0.0, so every program — including hall-of-fame
programs evolved before the context layer existed — remains evaluable everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features.context import CONTEXT_FEATURE_NAMES

BASE_FEATURE_NAMES: tuple[str, ...] = ("close", "ret", "logret", "hl_range", "volume_z")

FEATURE_NAMES: tuple[str, ...] = (*BASE_FEATURE_NAMES, *CONTEXT_FEATURE_NAMES)


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

    features = {
        "close": close,
        "ret": ret,
        "logret": logret,
        "hl_range": hl_range,
        "volume_z": volume_z,
    }
    for name in CONTEXT_FEATURE_NAMES:
        if name in bars.columns:
            features[name] = bars[name].astype(float).reset_index(drop=True).fillna(0.0)
        else:
            features[name] = pd.Series(0.0, index=close.index)
    return features
