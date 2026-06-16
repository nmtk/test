import numpy as np
import pandas as pd

from engine.strategies import TimeSeriesMomentum
from engine.validation.cpcv import cpcv_evaluate_strategy, cpcv_sharpes


def test_cpcv_combination_count_and_distribution():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 400))  # small positive drift
    res = cpcv_sharpes(returns, n_groups=6, n_test_groups=2, ppy=365.0)
    assert res.n_combinations == 15  # C(6, 2)
    assert len(res.combination_sharpes) == 15
    assert res.std > 0.0  # different sub-periods => a real distribution
    assert res.median > 0.0


def test_cpcv_on_strategy_runs():
    n = 500
    close = pd.Series(100.0 * (1.0005 ** np.arange(n)))  # gentle uptrend
    bars = pd.DataFrame(
        {"ts": range(n), "open": close, "high": close, "low": close, "close": close, "volume": 1.0}
    )
    res = cpcv_evaluate_strategy(bars, TimeSeriesMomentum(lookback=20), ppy=365.0)
    assert res.n_combinations == 15
    assert res.pct_positive == 1.0  # trend-follower in a steady uptrend
