import numpy as np
import pandas as pd
import pytest

from engine.indicators.core import (
    atr,
    donchian_high,
    donchian_low,
    returns,
    rolling_zscore,
    rsi,
    sma,
)


def test_sma_window():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_returns():
    s = pd.Series([100, 110, 99], dtype=float)
    r = returns(s)
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(-0.10)


def test_rolling_zscore():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    z = rolling_zscore(s, 3)
    assert z.iloc[2] == pytest.approx((3 - 2) / np.sqrt(2 / 3))


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 40), dtype=float)
    down = pd.Series(np.arange(40, 1, -1), dtype=float)
    assert rsi(up).dropna().eq(100.0).all()
    assert rsi(down).dropna().eq(0.0).all()
    mixed = pd.Series(np.sin(np.arange(60)) * 5 + 100)
    vals = rsi(mixed).dropna()
    assert ((vals >= 0) & (vals <= 100)).all()


def test_atr_positive():
    n = 30
    high = pd.Series(np.linspace(10, 20, n) + 1)
    low = pd.Series(np.linspace(10, 20, n) - 1)
    close = pd.Series(np.linspace(10, 20, n))
    a = atr(high, low, close, 14).dropna()
    assert (a > 0).all()


def test_donchian_excludes_current_bar():
    high = pd.Series([1, 2, 3, 4], dtype=float)
    low = pd.Series([1, 2, 3, 4], dtype=float)
    dh = donchian_high(high, 2)
    dl = donchian_low(low, 2)
    assert dh.iloc[2] == pytest.approx(2.0)  # max(high[0], high[1])
    assert dh.iloc[3] == pytest.approx(3.0)  # max(high[1], high[2])
    assert dl.iloc[2] == pytest.approx(1.0)
