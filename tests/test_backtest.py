import pandas as pd
import pytest

from engine.backtest import run_backtest, run_backtest_panel


def _s(values):
    return pd.Series(values, dtype=float)


def test_constant_long_on_uptrend_known_equity():
    close = _s([100, 110, 121, 133.1])  # +10% each step
    weights = _s([1, 1, 1, 1])
    r = run_backtest(close, weights)
    # applied weight is shifted, so equity = close[-1]/close[0]
    assert r.equity.iloc[-1] == pytest.approx(1.331)
    assert r.stats["total_return"] == pytest.approx(0.331)
    assert r.stats["max_drawdown"] == pytest.approx(0.0)
    assert r.stats["turnover"] == pytest.approx(1.0)  # one entry, then held


def test_cost_is_charged_on_turnover():
    close = _s([100, 110, 121, 133.1])
    weights = _s([1, 1, 1, 1])
    r = run_backtest(close, weights, fee_bps=100.0)  # 1% on the single entry
    # 1.09 * 1.1 * 1.1
    assert r.equity.iloc[-1] == pytest.approx(1.3189)


def test_no_look_ahead_last_weight_earns_nothing():
    close = _s([100, 110, 121])
    weights = _s([0, 0, 1])  # only acts on the final bar -> no future bar to earn from
    r = run_backtest(close, weights)
    assert r.stats["total_return"] == pytest.approx(0.0)


def test_flat_strategy_is_zero():
    close = _s([100, 110, 99, 105])
    weights = _s([0, 0, 0, 0])
    r = run_backtest(close, weights)
    assert r.stats["sharpe"] == 0.0
    assert (r.equity == 1.0).all()


def test_short_profits_on_downtrend():
    close = _s([100, 90, 81, 72.9])  # -10% each step
    weights = _s([-1, -1, -1, -1])
    r = run_backtest(close, weights)
    assert r.stats["total_return"] > 0.0


def test_panel_known_solution():
    closes = pd.DataFrame(
        {
            "A": [100, 110, 121, 133.1],  # +10%/bar
            "B": [100, 90, 81, 72.9],  # -10%/bar
        },
        dtype=float,
    )
    weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=closes.index)
    r = run_backtest_panel(closes, weights)
    # long winner + short loser => +10%/bar => same equity as a single +10% asset
    assert r.equity.iloc[-1] == pytest.approx(1.331)
    assert r.stats["turnover"] == pytest.approx(1.0)
