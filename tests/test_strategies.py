import numpy as np
import pandas as pd
import pytest

from engine.strategies import all_strategies, get_strategy


def _bars(close, high=None, low=None):
    close = pd.Series(close, dtype=float)
    return pd.DataFrame(
        {
            "ts": range(len(close)),
            "open": close,
            "high": close if high is None else pd.Series(high, dtype=float),
            "low": close if low is None else pd.Series(low, dtype=float),
            "close": close,
            "volume": 1.0,
        }
    )


def _rising(n=300):
    return _bars(list(100.0 * (1.01 ** np.arange(n))))


def _falling(n=300):
    return _bars(list(100.0 * (0.99 ** np.arange(n))))


# "gp" requires an evolved program argument; it is covered in the GP test modules.
@pytest.mark.parametrize("name", [n for n in all_strategies() if n != "gp"])
def test_weights_are_bounded_and_aligned(name):
    bars = _rising()
    w = get_strategy(name)().weights(bars)
    assert len(w) == len(bars)
    assert w.notna().all()
    assert (w.abs() <= 1.0 + 1e-9).all()


def test_tsmom_follows_trend():
    assert get_strategy("tsmom")().weights(_rising()).iloc[-1] == pytest.approx(1.0)
    assert get_strategy("tsmom")().weights(_falling()).iloc[-1] == pytest.approx(-1.0)


def test_ma_cross_follows_trend():
    assert get_strategy("ma_cross")().weights(_rising()).iloc[-1] == pytest.approx(1.0)
    assert get_strategy("ma_cross")().weights(_falling()).iloc[-1] == pytest.approx(-1.0)


def test_donchian_breaks_long_on_new_highs():
    assert get_strategy("donchian")().weights(_rising()).iloc[-1] == pytest.approx(1.0)


def test_zscore_reversion_fades_a_spike():
    close = [100.0] * 30 + [130.0]  # sharp jump up on the last bar
    w = get_strategy("zscore_reversion")(window=20, entry=1.0).weights(_bars(close))
    assert w.iloc[-1] < 0.0  # short the spike
