"""Carry archetype — hold the higher-yielding currency and collect the difference.

The most documented anomaly in foreign exchange, and the one this engine had no way to
express until now. It is also the one with the most distinctive shape: carry pays a
small, steady amount for long stretches and then gives back years of it in a few days
when the trade unwinds. That asymmetry is the point. A strategy whose returns look like
this is *not* interchangeable with a momentum or mean-reversion archetype even when its
average return matches, and a research engine that owns no carry hypothesis is blind to
a whole shape of risk rather than merely one more signal.

For USD/JPY the gap is the US yield minus Japan's, so a positive reading is
dollar-supportive. Japan held the world's lowest policy rate for most of three decades,
which is exactly why the yen became the funding currency of choice — and why yen carry
unwinds are the textbook example of the crowded-trade reversal.

Requires the ``mac_carry_*`` columns from :mod:`engine.features.macro_trade`; without
them the strategy reads neutral and stays flat, like every other conditioned archetype
here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.indicators.core import sma
from engine.strategies.base import FloatParam, IntParam, Strategy, register

SIGNAL_COLUMNS: dict[str, str] = {
    # The gap itself — what the trade earns while it is on.
    "level": "mac_carry_level",
    # Whether the gap is widening or closing, which is what moves the price.
    "momentum": "mac_carry_mom",
}


@register
class RateCarry(Strategy):
    """Lean toward the higher-yielding currency while the differential supports it.

    ``signal="level"`` trades the classic form: be long the high-yielder, size by how
    wide the gap is. ``signal="momentum"`` trades the gap's *direction* instead, which is
    a different bet — it can be short the carry while the carry is still positive, if the
    gap is closing.

    ``trend_filter`` is the concession to how carry actually fails. The unwind is fast
    and the fundamental says nothing about it; requiring price to agree gets the strategy
    out of the way of the crash that the rate differential will keep endorsing right
    through. The search decides whether that protection is worth what it costs in
    missed carry.
    """

    name = "carry"
    param_space = {
        "signal": list(SIGNAL_COLUMNS),
        "direction": [1, -1],
        "threshold": FloatParam(0.0, 0.6),
        "smooth": IntParam(1, 60),
        "trend_filter": [0, 1],
        "trend_window": IntParam(20, 200),
        "scale_by_level": [0, 1],
    }

    def __init__(
        self,
        signal: str = "level",
        direction: int = 1,
        threshold: float = 0.1,
        smooth: int = 5,
        trend_filter: int = 1,
        trend_window: int = 100,
        scale_by_level: int = 0,
    ) -> None:
        if signal not in SIGNAL_COLUMNS:
            raise ValueError(f"unknown signal {signal!r}; expected one of {sorted(SIGNAL_COLUMNS)}")
        self.signal = signal
        self.direction = int(direction)
        self.threshold = float(threshold)
        self.smooth = int(smooth)
        self.trend_filter = int(trend_filter)
        self.trend_window = int(trend_window)
        self.scale_by_level = int(scale_by_level)

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        column = SIGNAL_COLUMNS[self.signal]
        if column not in bars.columns:
            return pd.Series(0.0, index=bars.index)

        raw = bars[column].astype(float).fillna(0.0) * self.direction
        if self.smooth > 1:
            raw = raw.rolling(self.smooth, min_periods=1).mean()

        active = raw.abs() > self.threshold
        if self.scale_by_level:
            # Size by the width of the gap rather than taking a fixed position: a
            # half-point of carry is not the same trade as three points of it.
            weights = raw.clip(-1.0, 1.0).where(active, 0.0)
        else:
            weights = pd.Series(np.sign(raw), index=bars.index).where(active, 0.0)

        if self.trend_filter:
            close = bars["close"].astype(float)
            trend = np.sign(close - sma(close, self.trend_window)).fillna(0.0)
            weights = weights.where(np.sign(weights) == trend, 0.0)

        return weights.astype(float).fillna(0.0)


__all__ = ["RateCarry", "SIGNAL_COLUMNS"]
