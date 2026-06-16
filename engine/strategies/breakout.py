"""Donchian channel breakout — classic trend-entry archetype."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.indicators.core import donchian_high, donchian_low
from engine.strategies.base import IntParam, Strategy, register


@register
class DonchianBreakout(Strategy):
    """Long on a break above the prior ``window``-bar high, short on a break below the low.

    Position is held (forward-filled) until the opposite break occurs.
    """

    name = "donchian"
    param_space = {"window": IntParam(10, 100)}

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        upper = donchian_high(bars["high"], self.window)
        lower = donchian_low(bars["low"], self.window)

        signal = pd.Series(np.nan, index=close.index)
        signal[close > upper] = 1.0
        signal[close < lower] = -1.0
        # Hold the last breakout state; flat until the first signal.
        return signal.ffill().fillna(0.0).astype(float)
