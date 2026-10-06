"""The search objective: robust out-of-sample score, not raw backtest Sharpe.

Optimizing the in-sample Sharpe is how searches overfit. Instead we score a parameter set
by its **CPCV median Sharpe** (already cost-adjusted by the backtest) and subtract a
turnover penalty, so the optimizer prefers edges that survive out-of-sample and don't rely
on hyperactive trading. The independent gate still has the final say.
"""

from __future__ import annotations

from engine.backtest import backtest_strategy
from engine.validation.cpcv import cpcv_sharpes

DEFAULT_TURNOVER_PENALTY = 0.25


def robust_score(
    bars,
    strategy_cls,
    params: dict,
    *,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    ppy: float = 365.0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    cpcv_groups: int = 6,
    cpcv_test_groups: int = 2,
) -> float:
    result = backtest_strategy(
        strategy_cls(**params),
        bars,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        periods_per_year=ppy,
    )
    cpcv = cpcv_sharpes(
        result.returns, n_groups=cpcv_groups, n_test_groups=cpcv_test_groups, ppy=ppy
    )
    n_bars = max(int(result.stats["n_bars"]), 1)
    avg_turnover = result.stats["turnover"] / n_bars
    return float(cpcv.median - turnover_penalty * avg_turnover)
