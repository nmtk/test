"""The validation gate: run a candidate family through CPCV + PBO + DSR + SPA and decide.

A family is a set of configurations (e.g. one archetype's parameter grid). The gate selects
the in-sample best, then asks the hard questions: is the in-sample winner robust out of
sample (CPCV), is the *selection* overfit (PBO), is the winner significant after deflating
for the number of trials (DSR), and does the family beat the benchmark after multiple-
testing correction (SPA)? Only a candidate that clears every threshold becomes a challenger.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from engine.backtest import backtest_strategy
from engine.backtest.metrics import max_drawdown
from engine.validation.cpcv import cpcv_sharpes
from engine.validation.ledger import TrialLedger
from engine.validation.pbo import probability_of_backtest_overfitting
from engine.validation.spa import spa_test
from engine.validation.stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    return_moments,
)


@dataclass(frozen=True, slots=True)
class GateThresholds:
    dsr_min: float = 0.95
    pbo_max: float = 0.2
    spa_max: float = 0.05
    min_oos_sharpe: float = 0.0
    min_pct_positive: float = 0.5


@dataclass(slots=True)
class ValidationReport:
    name: str
    best_params: dict
    n_bars: int
    n_configs: int
    n_trials: int
    sharpe_annual: float
    cpcv_median: float
    cpcv_pct_positive: float
    psr: float
    dsr: float
    mintrl: float
    pbo: float
    spa_p: float
    # Backtest reference stats (per-bar) the incubation/decay bands compare forward data to.
    sharpe_per_bar: float = 0.0
    mean_per_bar: float = 0.0
    std_per_bar: float = 0.0
    skew: float = 0.0
    kurtosis: float = 3.0
    max_drawdown: float = 0.0
    checks: dict[str, bool] = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def returns_for_configs(
    bars: pd.DataFrame,
    strategy_cls,
    configs: list[dict],
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    ppy: float = 365.0,
) -> np.ndarray:
    """Backtest each config and stack per-bar returns into a (T x C) matrix."""
    columns = []
    for cfg in configs:
        result = backtest_strategy(
            strategy_cls(**cfg),
            bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            periods_per_year=ppy,
        )
        columns.append(result.returns.to_numpy(dtype=float))
    return np.column_stack(columns)


def build_returns_matrix(
    bars: pd.DataFrame,
    strategy_cls,
    param_grid: dict[str, list],
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    ppy: float = 365.0,
) -> tuple[np.ndarray, list[dict]]:
    """Backtest every parameter combination; return a (T x C) returns matrix and configs."""
    keys = list(param_grid)
    configs = [
        dict(zip(keys, values, strict=True)) for values in itertools.product(*param_grid.values())
    ]
    matrix = returns_for_configs(
        bars, strategy_cls, configs, fee_bps=fee_bps, slippage_bps=slippage_bps, ppy=ppy
    )
    return matrix, configs


def evaluate_matrix(
    returns_matrix: np.ndarray,
    *,
    name: str = "candidate",
    best_params: dict | None = None,
    ppy: float = 365.0,
    thresholds: GateThresholds | None = None,
    n_trials: int | None = None,
    n_partitions: int = 10,
    cpcv_groups: int = 6,
    cpcv_test_groups: int = 2,
    spa_boot: int = 500,
    spa_seed: int = 0,
) -> ValidationReport:
    """Run the full gate over a (T x C) returns matrix and return a report."""
    thresholds = thresholds or GateThresholds()
    matrix = np.asarray(returns_matrix, dtype=float)
    t, c = matrix.shape

    # In-sample winner by per-bar Sharpe.
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_bar_sharpe = np.where(std > 0, mean / std, 0.0)
    best = int(np.argmax(per_bar_sharpe))
    best_returns = pd.Series(matrix[:, best])

    moments = return_moments(best_returns)
    sharpe_variance = float(np.var(per_bar_sharpe, ddof=1)) if c > 1 else 0.0
    trials = n_trials if n_trials is not None else c

    psr = probabilistic_sharpe_ratio(moments.sharpe, moments.n, moments.skew, moments.kurtosis)
    dsr = deflated_sharpe_ratio(
        moments.sharpe, moments.n, moments.skew, moments.kurtosis, sharpe_variance, trials
    )
    sr_star = expected_max_sharpe(sharpe_variance, trials)
    mintrl = min_track_record_length(moments.sharpe, moments.skew, moments.kurtosis, sr_star)

    # PBO needs a family (>= 2 configs); a single config's selection can't be assessed, so
    # treat it conservatively as fully overfit rather than crashing.
    pbo = (
        probability_of_backtest_overfitting(matrix, n_partitions=n_partitions).pbo
        if c >= 2
        else 1.0
    )
    cpcv = cpcv_sharpes(best_returns, n_groups=cpcv_groups, n_test_groups=cpcv_test_groups, ppy=ppy)
    spa_p = spa_test(matrix, n_boot=spa_boot, seed=spa_seed).p_value

    checks = {
        "dsr": dsr >= thresholds.dsr_min,
        "pbo": pbo <= thresholds.pbo_max,
        "spa": spa_p <= thresholds.spa_max,
        "cpcv_median": cpcv.median >= thresholds.min_oos_sharpe,
        "cpcv_pct_positive": cpcv.pct_positive >= thresholds.min_pct_positive,
    }
    return ValidationReport(
        name=name,
        best_params=best_params or {},
        n_bars=t,
        n_configs=c,
        n_trials=trials,
        sharpe_annual=float(moments.sharpe * math.sqrt(ppy)),
        cpcv_median=cpcv.median,
        cpcv_pct_positive=cpcv.pct_positive,
        psr=psr,
        dsr=dsr,
        mintrl=mintrl,
        pbo=pbo,
        spa_p=spa_p,
        sharpe_per_bar=moments.sharpe,
        mean_per_bar=moments.mean,
        std_per_bar=moments.std,
        skew=moments.skew,
        kurtosis=moments.kurtosis,
        max_drawdown=max_drawdown((1.0 + best_returns).cumprod()),
        checks=checks,
        passed=all(checks.values()),
    )


def evaluate_candidate_family(
    bars: pd.DataFrame,
    strategy_cls,
    param_grid: dict[str, list],
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    ppy: float = 365.0,
    thresholds: GateThresholds | None = None,
    ledger: TrialLedger | None = None,
    family: str | None = None,
    **kwargs,
) -> ValidationReport:
    """Evaluate one archetype's parameter grid on ``bars``, updating the trial ledger.

    The DSR is deflated by the cumulative trial count for ``family`` (from the ledger) so
    repeated nightly searches are honestly penalized, not just tonight's grid.
    """
    matrix, configs = build_returns_matrix(
        bars, strategy_cls, param_grid, fee_bps=fee_bps, slippage_bps=slippage_bps, ppy=ppy
    )
    fam = family or getattr(strategy_cls, "name", "global")
    trials = len(configs)
    if ledger is not None:
        trials = ledger.add(fam, len(configs))

    # Identify best params for the report (same selection rule as evaluate_matrix).
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_bar_sharpe = np.where(std > 0, mean / std, 0.0)
    best_params = configs[int(np.argmax(per_bar_sharpe))]

    return evaluate_matrix(
        matrix,
        name=fam,
        best_params=best_params,
        ppy=ppy,
        thresholds=thresholds,
        n_trials=trials,
        **kwargs,
    )
