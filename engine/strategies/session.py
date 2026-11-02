"""Session archetype — the hypothesis that *when* a bar happens matters.

Every other archetype in this engine reads price or a macro level and asks which way to
lean. None of them ask what time it is, and that is a whole family of documented
structure going unexamined:

* FX liquidity and participant mix change completely between the Tokyo, London and New
  York sessions. The London open is the single most-studied intraday volatility event in
  the majors, and USD/JPY additionally has the 09:55 JST fixing, where corporate demand
  is concentrated into a few minutes by convention rather than by information.
* Crypto never closes, which makes weekends genuinely different: the same tape, with the
  institutional half of the participants absent.

This archetype tests the plainest version of that: **does a given window of the day carry
persistent drift?** Hold a fixed direction inside the window, flat outside it. If a
session has a real tilt, that shows up as a positive expectancy; if it does not, the
strategy is flat most of the time and earns nothing, which the gate will reject.

Deliberately simple. The point is to find out whether time-of-day carries signal at all
before building anything that leans on it — an hour-of-day filter bolted onto a
directional model would confound the two and tell us nothing about either.

On daily (or coarser) bars every bar falls in the same window, so the signal degenerates
to buy-and-hold. ``weights`` returns flat in that case rather than quietly reporting the
market's own return as the strategy's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.indicators.core import sma
from engine.strategies.base import IntParam, Strategy, register

# Below this bar size a day holds too few bars for a window to mean anything.
MIN_BARS_PER_DAY = 4


def hour_of_day(ts: pd.Series) -> pd.Series:
    """UTC hour (0-23) for each bar's open time."""
    return pd.to_datetime(ts.astype("int64"), unit="ms", utc=True).dt.hour


def day_of_week(ts: pd.Series) -> pd.Series:
    """Monday=0 ... Sunday=6, UTC."""
    return pd.to_datetime(ts.astype("int64"), unit="ms", utc=True).dt.dayofweek


def in_window(hours: pd.Series, start: int, end: int) -> pd.Series:
    """Hours inside ``[start, end)`` UTC, wrapping past midnight when ``start > end``."""
    if start == end:
        return pd.Series(False, index=hours.index)
    if start < end:
        return (hours >= start) & (hours < end)
    return (hours >= start) | (hours < end)  # e.g. Tokyo 23:00 -> 08:00


@register
class SessionDrift(Strategy):
    """Hold ``direction`` during a UTC window, flat outside it.

    ``start_hour``/``end_hour`` define the window, wrapping past midnight so an Asian
    session can be expressed. ``weekdays_only`` drops Saturday and Sunday, which matters
    for crypto (FX has no weekend bars to drop) and separates "this session drifts" from
    "the weekend drifts".

    ``trend_window`` optionally requires the longer-run trend to agree, so the search can
    distinguish an unconditional session tilt from one that only appears in an uptrend.
    Set it to 0 to trade the session unconditionally.
    """

    name = "session"
    param_space = {
        "start_hour": IntParam(0, 23),
        "end_hour": IntParam(0, 23),
        "direction": [1, -1],
        "weekdays_only": [0, 1],
        "trend_window": IntParam(0, 200),
    }

    def __init__(
        self,
        start_hour: int = 7,
        end_hour: int = 16,
        direction: int = 1,
        weekdays_only: int = 0,
        trend_window: int = 0,
    ) -> None:
        self.start_hour = int(start_hour) % 24
        self.end_hour = int(end_hour) % 24
        self.direction = int(direction)
        self.weekdays_only = int(weekdays_only)
        self.trend_window = int(trend_window)

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        flat = pd.Series(0.0, index=bars.index)
        if len(bars) < 2:
            return flat

        # A window only means something when a day holds several bars. On daily bars this
        # would otherwise collapse to buy-and-hold wearing a session's name.
        step_ms = int(bars["ts"].diff().median())
        if step_ms <= 0 or (86_400_000 / step_ms) < MIN_BARS_PER_DAY:
            return flat

        hours = hour_of_day(bars["ts"])
        active = in_window(hours, self.start_hour, self.end_hour)
        if self.weekdays_only:
            active &= day_of_week(bars["ts"]) < 5

        weights = pd.Series(float(self.direction), index=bars.index).where(active, 0.0)

        if self.trend_window > 0:
            close = bars["close"].astype(float)
            trend = np.sign(close - sma(close, self.trend_window)).fillna(0.0)
            weights = weights.where(np.sign(weights) == trend, 0.0)

        return weights.astype(float)


__all__ = ["SessionDrift", "hour_of_day", "day_of_week", "in_window"]
