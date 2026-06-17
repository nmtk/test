"""Fitness for GP programs: robust OOS score with complexity and turnover penalties.

Mirrors the parameter-search objective (CPCV median Sharpe, cost-adjusted) but adds a
size penalty so evolution prefers parsimonious formulas — smaller programs overfit less and
are auditable. The independent validation gate still has the final say.
"""

from __future__ import annotations

import math

import pandas as pd

from engine.backtest import run_backtest
from engine.gp.program import Node, evaluate, size, to_weight
from engine.validation.cpcv import cpcv_sharpes

_DEGENERATE = -1e9


def program_weights(node: Node, features: dict[str, pd.Series]) -> pd.Series:
    return to_weight(evaluate(node, features))


def fitness(
    node: Node,
    bars: pd.DataFrame,
    features: dict[str, pd.Series],
    *,
    ppy: float = 365.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    complexity_penalty: float = 0.002,
    turnover_penalty: float = 0.25,
    cpcv_groups: int = 6,
    cpcv_test_groups: int = 2,
) -> float:
    weights = program_weights(node, features)
    if float(weights.abs().sum()) == 0.0:
        return _DEGENERATE  # never trades -> useless
    result = run_backtest(
        bars["close"], weights, fee_bps=fee_bps, slippage_bps=slippage_bps, periods_per_year=ppy
    )
    cpcv = cpcv_sharpes(
        result.returns, n_groups=cpcv_groups, n_test_groups=cpcv_test_groups, ppy=ppy
    )
    if not math.isfinite(cpcv.median):
        return _DEGENERATE
    avg_turnover = result.stats["turnover"] / max(int(result.stats["n_bars"]), 1)
    return float(cpcv.median - complexity_penalty * size(node) - turnover_penalty * avg_turnover)
