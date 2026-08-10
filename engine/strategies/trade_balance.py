"""USD/JPY positioning from Japan-vs-US trade-flow fundamentals.

Trades the ``mac_*`` features built by :mod:`engine.features.macro_trade`: slow, real,
non-speculative currency demand created by the two countries' import/export performance.
Because trade statistics are published monthly with a multi-week lag, the signal changes a
dozen or so times a year — this is a fundamental tilt, not a timing model, and its natural
holding period is weeks.

Requires the ``mac_*`` columns; without them every signal reads neutral and the strategy
stays flat (safe by construction, exactly like ``macro_flow``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features.macro_trade import TRADE_FEATURE_NAMES
from engine.indicators.core import sma
from engine.strategies.base import FloatParam, IntParam, Strategy, register

# Signal name -> mac_ column. All bounded to ~[-1, 1], so one threshold scale fits each.
SIGNAL_COLUMNS: dict[str, str] = {
    "balance_diff": "mac_balance_diff",  # US balance improving vs Japan's -> USD/JPY up
    "jp_balance": "mac_jp_balance",  # Japan's balance alone (yen-side flow)
    "us_balance": "mac_us_balance",  # the US balance alone (dollar-side flow)
    "jp_import_cost": "mac_jp_import_cost",  # Japan's import bill (the energy channel)
}


@register
class TradeBalanceUsdJpy(Strategy):
    """Hold USD/JPY in the direction implied by the trade-flow signal.

    ``direction=+1`` trades the economically motivated sign — a US trade position
    improving relative to Japan's supports the dollar against the yen, so USD/JPY rises.
    ``-1`` trades the inverse, because "the flow story is already in the price" is an
    equally testable hypothesis and the validation gate, not the author, should decide
    which one survives out of sample.

    ``smooth`` averages the raw signal over N bars so a single revised print cannot flip
    the whole book; ``trend_filter`` additionally requires price to agree with the
    fundamental, which keeps the strategy out of the long stretches where a real but slow
    flow is being overwhelmed by rate differentials.
    """

    name = "trade_balance"
    param_space = {
        "signal": list(SIGNAL_COLUMNS),
        "direction": [1, -1],
        "threshold": FloatParam(0.0, 0.6),
        "smooth": IntParam(1, 60),
        "trend_filter": [0, 1],
        "trend_window": IntParam(20, 200),
    }

    def __init__(
        self,
        signal: str = "balance_diff",
        direction: int = 1,
        threshold: float = 0.1,
        smooth: int = 5,
        trend_filter: int = 0,
        trend_window: int = 100,
    ) -> None:
        if signal not in SIGNAL_COLUMNS:
            raise ValueError(f"unknown signal {signal!r}; expected one of {sorted(SIGNAL_COLUMNS)}")
        self.signal = signal
        self.direction = int(direction)
        self.threshold = float(threshold)
        self.smooth = int(smooth)
        self.trend_filter = int(trend_filter)
        self.trend_window = int(trend_window)

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        column = SIGNAL_COLUMNS[self.signal]
        if column not in bars.columns:
            # No trade-flow features stamped on these bars: stay flat rather than trade a
            # fundamental the validation never approved.
            return pd.Series(0.0, index=bars.index)

        raw = bars[column].astype(float).fillna(0.0) * self.direction
        if self.smooth > 1:
            raw = raw.rolling(self.smooth, min_periods=1).mean()

        weights = np.sign(raw) * (raw.abs() > self.threshold)

        if self.trend_filter:
            close = bars["close"].astype(float)
            trend = np.sign(close - sma(close, self.trend_window)).fillna(0.0)
            # Only hold where price and the fundamental point the same way.
            weights = weights.where(np.sign(weights) == trend, 0.0)

        return weights.astype(float).fillna(0.0)


__all__ = ["TradeBalanceUsdJpy", "SIGNAL_COLUMNS", "TRADE_FEATURE_NAMES"]
