"""Sharpe-ratio statistics: PSR, Deflated Sharpe, expected-max-Sharpe, MinTRL.

All Sharpe ratios here are **per-bar** (mean/std of the return series, not annualized) so
the small-sample formulas of Bailey & López de Prado apply directly. Annualization is a
reporting concern handled elsewhere. The normal CDF/inverse come from the stdlib
``statistics.NormalDist`` (no scipy dependency).

References: Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier" (2012) and
"The Deflated Sharpe Ratio" (2014).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True, slots=True)
class ReturnMoments:
    n: int
    sharpe: float  # per-bar
    skew: float
    kurtosis: float  # non-excess (normal == 3)


def return_moments(returns: pd.Series | np.ndarray) -> ReturnMoments:
    r = pd.Series(returns).dropna()
    n = int(len(r))
    if n < 2:
        return ReturnMoments(n=n, sharpe=0.0, skew=0.0, kurtosis=3.0)
    sd = float(r.std(ddof=1))
    sharpe = 0.0 if sd == 0.0 else float(r.mean() / sd)
    skew = float(r.skew()) if n > 2 else 0.0
    # pandas .kurt() is *excess* (Fisher); the PSR/DSR formulas use non-excess kurtosis.
    excess_kurt = float(r.kurt()) if n > 3 else 0.0
    return ReturnMoments(n=n, sharpe=sharpe, skew=skew, kurtosis=excess_kurt + 3.0)


def _psr_denominator(sharpe: float, skew: float, kurtosis: float) -> float:
    """sqrt(1 - skew*SR + (kurt-1)/4 * SR^2); the variance of the Sharpe estimator."""
    inner = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    return math.sqrt(inner) if inner > 0.0 else float("nan")


def probabilistic_sharpe_ratio(
    sharpe: float, n: int, skew: float, kurtosis: float, benchmark_sharpe: float = 0.0
) -> float:
    """P(true Sharpe > benchmark) given the estimate's standard error (per-bar Sharpe)."""
    if n < 2:
        return 0.0
    denom = _psr_denominator(sharpe, skew, kurtosis)
    if not math.isfinite(denom) or denom == 0.0:
        return 0.0
    z = (sharpe - benchmark_sharpe) * math.sqrt(n - 1) / denom
    return float(_N.cdf(z))


def expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float:
    """Expected maximum (per-bar) Sharpe across ``n_trials`` independent strategies.

    The benchmark a winner must beat to be credible. Bailey & López de Prado (2014).
    """
    if n_trials < 2 or sharpe_variance <= 0.0:
        return 0.0
    sigma = math.sqrt(sharpe_variance)
    gamma = _EULER_MASCHERONI
    z1 = _N.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(sigma * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(
    sharpe: float,
    n: int,
    skew: float,
    kurtosis: float,
    sharpe_variance: float,
    n_trials: int,
) -> float:
    """PSR against the expected-maximum-Sharpe benchmark for ``n_trials`` (per-bar Sharpe).

    This is the headline overfitting-aware significance: high only if the strategy beats
    what the *best of N random trials* would have produced by luck.
    """
    sr_star = expected_max_sharpe(sharpe_variance, n_trials)
    return probabilistic_sharpe_ratio(sharpe, n, skew, kurtosis, benchmark_sharpe=sr_star)


def min_track_record_length(
    sharpe: float,
    skew: float,
    kurtosis: float,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Minimum number of observations for PSR(benchmark) to reach ``confidence``.

    Returns ``inf`` if the strategy does not beat the benchmark (PSR can never reach it).
    """
    if sharpe <= benchmark_sharpe:
        return float("inf")
    denom = _psr_denominator(sharpe, skew, kurtosis)
    if not math.isfinite(denom):
        return float("inf")
    z = _N.inv_cdf(confidence)
    return float(1.0 + (denom**2) * (z / (sharpe - benchmark_sharpe)) ** 2)
