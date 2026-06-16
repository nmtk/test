"""Core causal indicators used by the seed strategy archetypes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Simple returns over ``periods`` bars."""
    return close.pct_change(periods)


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close).diff(periods)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """(x - rolling mean) / rolling std, computed causally."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When there are no losses over the window, RSI saturates at 100.
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def donchian_high(high: pd.Series, window: int) -> pd.Series:
    """Highest high of the *prior* ``window`` bars (excludes the current bar)."""
    return high.shift(1).rolling(window, min_periods=window).max()


def donchian_low(low: pd.Series, window: int) -> pd.Series:
    """Lowest low of the *prior* ``window`` bars (excludes the current bar)."""
    return low.shift(1).rolling(window, min_periods=window).min()
