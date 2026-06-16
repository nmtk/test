"""Apply CPCV to a strategy's return stream to get a robust OOS Sharpe distribution.

Rather than trusting one backtest number, we evaluate the strategy on every combination of
test blocks and look at the *distribution* of out-of-sample Sharpe. The median is the
robust estimate; a wide spread or many negative combinations is a red flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.backtest import run_backtest
from engine.validation.splits import combinatorial_purged_splits


@dataclass(frozen=True, slots=True)
class CPCVResult:
    combination_sharpes: list[float]  # annualized Sharpe per test-block combination
    median: float
    mean: float
    std: float
    pct_positive: float
    n_combinations: int


def _annualized_sharpe(returns: np.ndarray, ppy: float) -> float:
    r = returns[~np.isnan(returns)]
    if len(r) < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(r) / sd * math.sqrt(ppy))


def cpcv_sharpes(
    returns: pd.Series,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_frac: float = 0.01,
    ppy: float = 365.0,
) -> CPCVResult:
    values = pd.Series(returns).to_numpy(dtype=float)
    splits = combinatorial_purged_splits(
        len(values), n_groups=n_groups, n_test_groups=n_test_groups, embargo_frac=embargo_frac
    )
    sharpes = [_annualized_sharpe(values[s.test_idx], ppy) for s in splits]
    arr = np.array(sharpes, dtype=float)
    return CPCVResult(
        combination_sharpes=sharpes,
        median=float(np.median(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        pct_positive=float(np.mean(arr > 0.0)),
        n_combinations=len(sharpes),
    )


def cpcv_evaluate_strategy(
    bars: pd.DataFrame,
    strategy,
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    ppy: float = 365.0,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_frac: float = 0.01,
) -> CPCVResult:
    weights = strategy.weights(bars)
    result = run_backtest(
        bars["close"], weights, fee_bps=fee_bps, slippage_bps=slippage_bps, periods_per_year=ppy
    )
    return cpcv_sharpes(
        result.returns,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        embargo_frac=embargo_frac,
        ppy=ppy,
    )
