"""Momentum / trend-following archetypes — the most robust cross-asset anomaly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.indicators.core import sma
from engine.strategies.base import IntParam, Strategy, register


@register
class TimeSeriesMomentum(Strategy):
    """Go long if the trailing ``lookback``-bar return is positive, short if negative.

    Moskowitz-Ooi-Pedersen (2012); Hurst et al. (2017). Long/short ±1.
    """

    name = "tsmom"
    param_space = {"lookback": IntParam(5, 200)}

    def __init__(self, lookback: int = 50) -> None:
        self.lookback = lookback

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        past = close.shift(self.lookback)
        signal = np.sign(close / past - 1.0)
        return signal.fillna(0.0).astype(float)


@register
class MovingAverageCross(Strategy):
    """Long when fast SMA is above slow SMA, short otherwise (±1)."""

    name = "ma_cross"
    param_space = {"fast": IntParam(5, 50), "slow": IntParam(20, 200)}

    def __init__(self, fast: int = 20, slow: int = 100) -> None:
        self.fast = fast
        self.slow = slow

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        fast = sma(close, self.fast)
        slow = sma(close, self.slow)
        signal = np.sign(fast - slow)
        return signal.fillna(0.0).astype(float)
