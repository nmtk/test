"""Probability of Backtest Overfitting via CSCV (Bailey et al., 2017).

Given a matrix of per-bar returns for many candidate configurations, CSCV repeatedly
splits the timeline in half, picks the configuration that looks best in-sample, and checks
where it ranks out-of-sample. If the in-sample winner is routinely a below-median performer
out-of-sample, the selection process is overfitting. PBO is the fraction of splits where
that happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True, slots=True)
class PBOResult:
    pbo: float
    logits: list[float]
    n_splits: int


def _sharpe_columns(matrix: np.ndarray) -> np.ndarray:
    """Per-column (config) per-bar Sharpe, NaN-safe."""
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0, mean / std, 0.0)
    return sharpe


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray, n_partitions: int = 10
) -> PBOResult:
    """CSCV PBO over a (T x C) returns matrix (T bars, C configurations).

    ``n_partitions`` (S, even) contiguous blocks are split into all C(S, S/2) in-sample /
    out-of-sample halves. Returns PBO in [0, 1]; lower is better (rule of thumb: < 0.2).
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("returns_matrix must be (T, C) with C >= 2 configurations")
    if n_partitions % 2 != 0 or n_partitions < 2:
        raise ValueError("n_partitions must be a positive even number")

    t = matrix.shape[0]
    if t < n_partitions:
        raise ValueError("not enough rows for the requested number of partitions")

    edges = np.linspace(0, t, n_partitions + 1).astype(int)
    blocks = [matrix[edges[i] : edges[i + 1]] for i in range(n_partitions)]

    logits: list[float] = []
    half = n_partitions // 2
    for is_combo in combinations(range(n_partitions), half):
        oos_combo = [i for i in range(n_partitions) if i not in is_combo]
        is_data = np.vstack([blocks[i] for i in is_combo])
        oos_data = np.vstack([blocks[i] for i in oos_combo])

        is_sharpe = _sharpe_columns(is_data)
        oos_sharpe = _sharpe_columns(oos_data)

        best = int(np.argmax(is_sharpe))
        # Relative rank of the IS-best config in the OOS Sharpe ordering, in (0, 1).
        c = oos_sharpe.shape[0]
        rank = float((np.argsort(np.argsort(oos_sharpe))[best] + 1) / (c + 1))
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1.0 - rank)))

    logit_arr = np.array(logits, dtype=float)
    # Overfit when the IS-best lands at or below the OOS median (logit <= 0).
    pbo = float(np.mean(logit_arr <= 0.0))
    return PBOResult(pbo=pbo, logits=logits, n_splits=len(logits))
