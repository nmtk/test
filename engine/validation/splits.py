"""Combinatorial purged cross-validation splits (López de Prado, 2018).

The timeline is cut into ``n_groups`` contiguous blocks; every combination of
``n_test_groups`` blocks forms a test set, the rest is train. Train samples whose label
window overlaps the test (``purge``) or fall just after a test block (``embargo``) are
dropped, so no information leaks across the boundary. This generalizes a single
walk-forward split into many overlapping OOS evaluations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True, slots=True)
class CPCVSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple[int, ...]


def n_backtest_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of complete OOS paths CPCV reconstructs: C(N,k) * k / N."""
    return math.comb(n_groups, n_test_groups) * n_test_groups // n_groups


def _group_bounds(n_samples: int, n_groups: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n_samples, n_groups + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def combinatorial_purged_splits(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_frac: float = 0.01,
    purge_gap: int = 1,
) -> list[CPCVSplit]:
    """Generate all CPCV splits for ``n_samples`` observations.

    ``purge_gap`` is the label horizon in bars (train samples within this many bars of a
    test block are purged). ``embargo_frac`` of total samples is embargoed after each test
    block. Raises if the configuration is degenerate.
    """
    if not (1 <= n_test_groups < n_groups):
        raise ValueError("require 1 <= n_test_groups < n_groups")
    if n_samples < n_groups:
        raise ValueError("n_samples must be >= n_groups")

    bounds = _group_bounds(n_samples, n_groups)
    embargo = int(round(embargo_frac * n_samples))
    all_idx = np.arange(n_samples)
    splits: list[CPCVSplit] = []

    for combo in combinations(range(n_groups), n_test_groups):
        test_mask = np.zeros(n_samples, dtype=bool)
        excluded = np.zeros(n_samples, dtype=bool)  # test + purge + embargo
        for g in combo:
            start, end = bounds[g]
            test_mask[start:end] = True
            lo = max(0, start - purge_gap)
            hi = min(n_samples, end + purge_gap + embargo)
            excluded[lo:hi] = True
        train_idx = all_idx[~excluded]
        test_idx = all_idx[test_mask]
        splits.append(CPCVSplit(train_idx=train_idx, test_idx=test_idx, test_groups=combo))
    return splits
