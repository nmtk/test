"""Hansen's Superior Predictive Ability test (2005), stationary-bootstrap variant.

When you compare many strategies against a benchmark and keep the best, the maximum
performance is upward-biased. SPA gives a p-value for the null "no strategy in the family
beats the benchmark", controlling for that data-snooping bias. We use the consistent
recentering (SPA_c) and a stationary bootstrap (Politis & Romano, 1994).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SPAResult:
    p_value: float
    statistic: float
    n_boot: int


def stationary_bootstrap_indices(
    n: int, avg_block: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot, n) index matrix for the stationary bootstrap with mean block ``avg_block``."""
    p = 1.0 / max(avg_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    for t in range(1, n):
        restart = rng.random(n_boot) < p
        cont = (idx[:, t - 1] + 1) % n
        fresh = rng.integers(0, n, size=n_boot)
        idx[:, t] = np.where(restart, fresh, cont)
    return idx


def spa_test(
    performance: np.ndarray,
    *,
    block_size: float = 10.0,
    n_boot: int = 1000,
    seed: int = 0,
) -> SPAResult:
    """SPA p-value for H0: max_k E[performance_k] <= 0.

    ``performance`` is a (T x C) matrix of per-bar performance of each candidate relative to
    the benchmark (positive == better than benchmark). A small p-value means at least one
    candidate genuinely beats the benchmark after accounting for multiple testing.
    """
    perf = np.asarray(performance, dtype=float)
    if perf.ndim != 2 or perf.shape[1] < 1:
        raise ValueError("performance must be (T, C) with C >= 1")
    t, c = perf.shape

    mean = np.nanmean(perf, axis=0)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(t, block_size, n_boot, rng)

    # Bootstrap means per config: (n_boot, C). Used for both omega and the null distribution.
    boot_means = np.stack([np.nanmean(perf[idx[b]], axis=0) for b in range(n_boot)], axis=0)
    omega = np.std(math.sqrt(t) * boot_means, axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)

    with np.errstate(invalid="ignore"):
        t_stats = math.sqrt(t) * mean / omega
    statistic = float(np.nanmax(np.append(t_stats, 0.0)))

    # Consistent recentering: only "good enough" configs contribute under the null.
    threshold = math.sqrt(2.0 * math.log(math.log(t))) if t > math.e else 0.0
    keep = t_stats >= -threshold
    g = np.where(keep, mean, 0.0)

    with np.errstate(invalid="ignore"):
        z = math.sqrt(t) * (boot_means - g) / omega  # (n_boot, C)
    z = np.nan_to_num(z, nan=-np.inf)
    boot_stat = np.maximum(np.max(z, axis=1), 0.0)

    p_value = float(np.mean(boot_stat > statistic))
    return SPAResult(p_value=p_value, statistic=statistic, n_boot=n_boot)
