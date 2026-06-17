"""Change-detection primitives used to spot strategy decay.

* :func:`page_hinkley` — one-sided Page-Hinkley test for a *decrease* in mean.
* :func:`cusum_down` — one-sided CUSUM for accumulation below a target.
* :func:`sharpe_lower_band` — one-sided lower confidence bound on a backtest Sharpe.

Inputs are typically *standardized* forward returns (z-scored against the backtest mean/std),
so thresholds are scale-free.
"""

from __future__ import annotations

import numpy as np

from engine.validation.stats import sharpe_std_error


def page_hinkley(values, delta: float = 0.5, threshold: float = 5.0) -> bool:
    """True if a downward shift in the mean of ``values`` is detected.

    Decrease-form Page-Hinkley: ``U_t = sum(x - running_mean + delta)`` drifts upward by the
    tolerance ``delta`` so ordinary noise does not trigger; a sustained drop makes
    ``max_s(U_s) - U_t`` exceed ``threshold``. Inputs are expected standardized (delta in σ).
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return False
    mean = 0.0
    u = 0.0
    u_max = 0.0
    for t, xi in enumerate(x, start=1):
        mean += (xi - mean) / t
        u += xi - mean + delta
        u_max = max(u_max, u)
        if u_max - u > threshold:
            return True
    return False


def cusum_down(values, target: float = 0.0, drift: float = 0.5, threshold: float = 5.0) -> bool:
    """True if a one-sided CUSUM accumulates evidence that values fall below ``target``.

    ``drift`` is the slack (in input units) below ``target`` tolerated before accumulating.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    s = 0.0
    for xi in x:
        s = max(0.0, s + (target - xi - drift))
        if s > threshold:
            return True
    return False


def sharpe_lower_band(
    sharpe: float, n: int, skew: float, kurtosis: float, z: float = 1.645
) -> float:
    """One-sided lower confidence bound on a (per-bar) backtest Sharpe.

    Forward performance below this band is evidence the edge was overstated or has decayed.
    """
    se = sharpe_std_error(sharpe, n, skew, kurtosis)
    if not np.isfinite(se):
        return -np.inf
    return sharpe - z * se
