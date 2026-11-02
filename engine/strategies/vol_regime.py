"""Volatility-regime archetype — betting on the volatility state, not the price level.

Every other archetype here reads a *price* and leans a direction. This one reads how
*agitated* the market is and acts on the transition.

The hypothesis is the oldest observation in the volatility literature: variance clusters
and cycles. Quiet begets quiet until it does not, and the breaking of a quiet stretch is
not the same event as the same move arriving mid-storm. A range break out of a six-month
coil and an identical-sized break during a panic look alike to ``donchian``, which sees
only the price range; they are different trades, and this archetype is the one that can
tell them apart.

Which way to trade the break is left open on purpose. Both readings are respectable and
both are observed in real markets:

* **continuation** — compression stores energy; the first expansion is information
  arriving, and information keeps arriving.
* **reversion** — a quiet market has thin books, so the first expansion is as likely to
  be a liquidity air pocket that snaps back as it is to be news.

``direction`` picks which is being tested, and the validation gate decides per instrument
and timeframe which (if either) survives out of sample. Asserting one of them here would
be putting the author's prior where the evidence should go.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.indicators.core import atr, true_range
from engine.strategies.base import FloatParam, IntParam, Strategy, register


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of log returns — the plain realized-volatility estimate."""
    logret = np.log(close.replace(0.0, np.nan)).diff()
    return logret.rolling(window, min_periods=window).std(ddof=0)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Where each value sits within its own trailing ``window``, in ``[0, 1]``.

    Causal: the rank of the newest point among the window ending at that point, so a
    reading of 0.05 means "quieter than 95% of the recent past, as known then".
    """
    return series.rolling(window, min_periods=window // 2).rank(pct=True)


@register
class VolRegimeBreak(Strategy):
    """Trade the first expansion out of a compressed-volatility stretch.

    A bar qualifies when both are true at its close:

    * realized volatility was in the quietest ``compress_pct`` of its trailing
      distribution as of the *previous* bar — the market was coiled before this bar, not
      because of it;
    * this bar's true range exceeds ``expansion_mult`` times the recent ATR — the coil
      broke.

    The position then follows ``direction * sign(bar return)`` and is held for
    ``max_hold`` bars or until volatility is no longer elevated, whichever comes first.
    """

    name = "vol_regime"
    param_space = {
        "vol_window": IntParam(10, 60),
        "rank_window": IntParam(50, 400),
        "compress_pct": FloatParam(0.05, 0.40),
        "expansion_mult": FloatParam(1.2, 4.0),
        "direction": [1, -1],
        "max_hold": IntParam(1, 30),
    }

    def __init__(
        self,
        vol_window: int = 20,
        rank_window: int = 200,
        compress_pct: float = 0.20,
        expansion_mult: float = 2.0,
        direction: int = 1,
        max_hold: int = 10,
    ) -> None:
        self.vol_window = int(vol_window)
        self.rank_window = int(rank_window)
        self.compress_pct = float(compress_pct)
        self.expansion_mult = float(expansion_mult)
        self.direction = int(direction)
        self.max_hold = int(max_hold)

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        high, low = bars["high"].astype(float), bars["low"].astype(float)
        n = len(bars)
        if n < self.rank_window + self.vol_window:
            return pd.Series(0.0, index=bars.index)

        vol = realized_vol(close, self.vol_window)
        # Shifted: the regime must have been quiet *before* this bar, or the expanding
        # bar would be judging the state it just created.
        compressed = (rolling_percentile(vol, self.rank_window) <= self.compress_pct).shift(1)

        reference = atr(high, low, close, self.vol_window).shift(1)
        expanded = true_range(high, low, close) > reference * self.expansion_mult

        trigger = (compressed & expanded).fillna(False).to_numpy()
        move = np.sign(close.pct_change().fillna(0.0).to_numpy())
        still_loud = (true_range(high, low, close) > reference).fillna(False).to_numpy()

        # Sequential: whether a bar opens a position depends on whether an earlier one is
        # still running, which no vectorized form expresses.
        position = np.zeros(n, dtype=float)
        side = 0.0
        held = 0
        for i in range(n):
            if side != 0.0:
                held += 1
                if held >= self.max_hold or not still_loud[i]:
                    side = 0.0
                    held = 0
            if side == 0.0 and trigger[i] and move[i] != 0.0:
                side = float(self.direction) * move[i]
                held = 0
            position[i] = side

        return pd.Series(position, index=bars.index, dtype=float)


__all__ = ["VolRegimeBreak", "realized_vol", "rolling_percentile"]
