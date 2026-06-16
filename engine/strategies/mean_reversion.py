"""Mean-reversion archetype — fade extreme deviations from a rolling mean."""

from __future__ import annotations

import pandas as pd

from engine.indicators.core import rolling_zscore
from engine.strategies.base import FloatParam, IntParam, Strategy, register


@register
class ZScoreReversion(Strategy):
    """Fade the rolling z-score of price: short when stretched up, long when stretched down.

    Enter only beyond ``entry`` standard deviations; weight scales with the z-score up to
    ±1. Cost-sensitive and regime-dependent (a liquidity-provision premium).
    """

    name = "zscore_reversion"
    param_space = {"window": IntParam(5, 100), "entry": FloatParam(0.5, 3.0)}

    def __init__(self, window: int = 20, entry: float = 1.0) -> None:
        self.window = window
        self.entry = entry

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        z = rolling_zscore(bars["close"], self.window)
        # Fade: weight = -z / (2*entry), clipped to [-1, 1]; flat inside the entry band.
        raw = (-z / (2.0 * self.entry)).clip(-1.0, 1.0)
        weights = raw.where(z.abs() >= self.entry, 0.0)
        return weights.fillna(0.0).astype(float)
