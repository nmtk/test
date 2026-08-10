"""WEMOF — "trade only the moves nothing explains" reversal archetype.

A mechanization of the discretionary scalping method popularized by わっきゃい (Wakkyai)
under the name *WEMOF* (Wakkyai's Entry Model Of FX), as described publicly in interviews
and write-ups. The published rules, on USD/JPY 5m:

1. Price touching Bollinger ±3σ is **entry preparation, not an entry**.
2. The touch is *skipped* when the move is **explainable** — a headline/news impulse, a
   break of a prior high/low, or a break of a key level (pivot, round number, triangle).
3. Only an **unexplained** overshoot is faded, counter-trend, for a small quick target.

Point 2 is the whole method. A plain ±3σ fade is a well-known, thoroughly crowded trade;
WEMOF's claim is that conditioning it on *the absence of an identifiable cause* separates
genuine statistical noise (which mean-reverts) from the start of a real repricing (which
does not). So this archetype implements the vetoes as first-class, individually switchable
filters and lets the optuna search + validation gate decide — per symbol and timeframe —
which of them actually carry out-of-sample weight, rather than asserting they all do.

Faithfulness notes (what is and is not reproducible in this engine):

* Only the *published, verbally described* rules are reproduced. No proprietary indicator
  code is used — this is a from-scratch implementation of a publicly documented method.
* The "news" veto in the original is a human reading the headline feed. Without a
  point-in-time economic-calendar source, this uses a **volatility-burst proxy**: an
  impulse whose true range dwarfs recent ATR is treated as explained. That is a
  *different* filter with the same intent, and it is labelled as such.
* Exits are evaluated on **bar closes only**, never intrabar. The original's ~1-2 pip
  take-profit against a ~3-8 pip stop lives inside a single 5m bar and cannot be resolved
  from OHLCV without tick data — modelling it from bar highs/lows would silently invent
  fills that may never have happened in that order. Close-based exits are coarser but
  unbiased, which is the trade this repo's validation gate is built to protect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.indicators.core import atr, donchian_high, donchian_low, sma
from engine.strategies.base import FloatParam, IntParam, Strategy, register

# Each veto answers "is this overshoot explainable?". Any veto that fires suppresses the
# entry. Exposed individually so the search can measure each one's marginal contribution.
VETO_NAMES: tuple[str, ...] = ("structure", "level", "news", "trend")


def bollinger(
    close: pd.Series, window: int, sigma: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(lower, middle, upper)`` Bollinger bands, computed causally."""
    middle = sma(close, window)
    width = close.rolling(window, min_periods=window).std(ddof=0) * float(sigma)
    return middle - width, middle, middle + width


def round_level_distance(close: pd.Series, decades: int) -> pd.Series:
    """Distance from each price to the nearest round-number level, in units of that grid.

    The grid step is chosen relative to the instrument's own price magnitude, so the same
    parameter means "whole yen" on USD/JPY at 150 and a proportionate level on BTC/JPY at
    10,000,000. ``decades`` counts digits below the leading one: for a price near 150,
    ``decades=2`` gives a step of 1.0 and ``decades=3`` a step of 0.1.

    Returns a value in ``[0, 0.5]`` — 0 means sitting exactly on a round number.
    """
    price = close.abs().replace(0.0, np.nan)
    magnitude = np.floor(np.log10(price))
    step = np.power(10.0, magnitude - float(decades))
    units = price / step
    return pd.Series(np.abs(units - np.round(units)), index=close.index).fillna(0.5)


@register
class WemofReversal(Strategy):
    """Fade a Bollinger ``sigma``-band overshoot, but only when no veto explains it.

    Entry (evaluated at each bar's close, on information available at that close):

    * the bar's low pierced the lower band -> *prepare* long; high pierced the upper band
      -> *prepare* short;
    * every enabled veto must be silent, otherwise the touch is skipped;
    * position is taken at full size (``±1``) and the engine applies it from the next bar.

    Exit, whichever comes first:

    * price closes back through the band's middle (the reversion the trade was betting on);
    * ``max_hold`` bars elapse (the original is a scalp — it does not sit in a losing
      position hoping);
    * close moves ``stop_atr`` ATRs against the entry.
    """

    name = "wemof"
    param_space = {
        "bb_window": IntParam(10, 40),
        "bb_sigma": FloatParam(1.8, 3.5),
        "max_hold": IntParam(1, 24),
        "stop_atr": FloatParam(0.5, 4.0),
        "veto_structure": [0, 1],
        "veto_level": [0, 1],
        "veto_news": [0, 1],
        "veto_trend": [0, 1],
        "structure_window": IntParam(10, 120),
        "level_decades": IntParam(2, 4),
        "level_tolerance": FloatParam(0.02, 0.25),
        "news_atr_mult": FloatParam(1.2, 4.0),
        "trend_window": IntParam(20, 200),
    }

    def __init__(
        self,
        bb_window: int = 20,
        bb_sigma: float = 3.0,
        max_hold: int = 6,
        stop_atr: float = 2.0,
        veto_structure: int = 1,
        veto_level: int = 1,
        veto_news: int = 1,
        veto_trend: int = 0,
        structure_window: int = 40,
        level_decades: int = 3,
        level_tolerance: float = 0.08,
        news_atr_mult: float = 2.0,
        trend_window: int = 100,
    ) -> None:
        self.bb_window = int(bb_window)
        self.bb_sigma = float(bb_sigma)
        self.max_hold = int(max_hold)
        self.stop_atr = float(stop_atr)
        self.veto_structure = int(veto_structure)
        self.veto_level = int(veto_level)
        self.veto_news = int(veto_news)
        self.veto_trend = int(veto_trend)
        self.structure_window = int(structure_window)
        self.level_decades = int(level_decades)
        self.level_tolerance = float(level_tolerance)
        self.news_atr_mult = float(news_atr_mult)
        self.trend_window = int(trend_window)

    # -- explainability vetoes ---------------------------------------------------------

    def _explained(self, bars: pd.DataFrame, side: int) -> pd.Series:
        """True where a ``side`` (+1 long / -1 short) overshoot has an identifiable cause.

        ``side`` is the *trade* direction, so a long candidate is a downside overshoot.
        """
        high, low, close = bars["high"], bars["low"], bars["close"]
        explained = pd.Series(False, index=bars.index)

        if self.veto_structure:
            # A break of the prior N-bar extreme is a structural event, not an anomaly:
            # the market is discovering a new level, so there is nothing to fade.
            # (Shifted by one bar so the bar cannot break a range it helped define.)
            if side > 0:
                broke = low < donchian_low(low, self.structure_window).shift(1)
            else:
                broke = high > donchian_high(high, self.structure_window).shift(1)
            explained |= broke.fillna(False)

        if self.veto_level:
            # Sitting on a round number: order flow clusters there, so an overshoot is
            # explained by stop runs / option barriers rather than by noise.
            near = round_level_distance(close, self.level_decades) < self.level_tolerance
            explained |= near.fillna(False)

        if self.veto_news:
            # Headline proxy: an impulse far larger than recent typical range. The original
            # reads the news feed; without a point-in-time calendar this stands in for it.
            true_range = (high - low).abs()
            reference = atr(high, low, close, self.bb_window).shift(1)
            burst = true_range > reference * self.news_atr_mult
            explained |= burst.fillna(False)

        if self.veto_trend:
            # A move that simply extends an established trend is explained by that trend.
            # Fading it is fighting the dominant flow, which the method avoids.
            trend = np.sign(close - sma(close, self.trend_window))
            explained |= (trend == -side).fillna(False)

        return explained

    # -- position construction ---------------------------------------------------------

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype(float)
        lower, middle, upper = bollinger(close, self.bb_window, self.bb_sigma)

        touch_long = (bars["low"].astype(float) < lower).fillna(False)
        touch_short = (bars["high"].astype(float) > upper).fillna(False)
        enter_long = touch_long & ~self._explained(bars, side=1)
        enter_short = touch_short & ~self._explained(bars, side=-1)

        band_ready = lower.notna() & middle.notna()
        atr_series = atr(bars["high"], bars["low"], close, self.bb_window)

        # Sequential because a scalp's state is path-dependent: whether bar t opens a
        # position depends on whether an earlier entry is still running.
        n = len(bars)
        position = np.zeros(n, dtype=float)
        long_sig = enter_long.to_numpy()
        short_sig = enter_short.to_numpy()
        ready = band_ready.to_numpy()
        px = close.to_numpy(dtype=float)
        mid = middle.to_numpy(dtype=float)
        atr_vals = atr_series.to_numpy(dtype=float)

        side = 0
        entry_px = 0.0
        entry_atr = 0.0
        held = 0
        for i in range(n):
            if side != 0:
                held += 1
                reverted = (side > 0 and px[i] >= mid[i]) or (side < 0 and px[i] <= mid[i])
                stopped = entry_atr > 0.0 and (entry_px - px[i]) * side >= self.stop_atr * entry_atr
                if reverted or stopped or held >= self.max_hold:
                    side = 0
                    entry_px = 0.0
                    entry_atr = 0.0
                    held = 0
            if side == 0 and ready[i]:
                if long_sig[i]:
                    side = 1
                elif short_sig[i]:
                    side = -1
                if side != 0:
                    entry_px = px[i]
                    entry_atr = atr_vals[i] if np.isfinite(atr_vals[i]) else 0.0
                    held = 0
            position[i] = float(side)

        return pd.Series(position, index=bars.index, dtype=float)
