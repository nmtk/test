"""Confidence-weighted, correlation-aware risk-parity allocation across champions.

For each champion we combine three deterministic factors:

* **risk parity** — weight ∝ 1 / volatility (don't let one loud strategy dominate);
* **confidence** — P(true mean return > 0) from the forward estimate (shrinks uncertain or
  losing strategies toward zero); recency is handled by the caller using a recent window;
* **correlation penalty** — down-weight strategies that move together (avoid double-betting
  the same idea).

Then hard caps are applied: a per-strategy cap and a tight cap on the whole **GP family**
(GP is experimental — see docs/MANUAL.md §5), and the book is normalized to the gross limit
(never forced up, so a cash buffer is fine).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

_N = NormalDist()


@dataclass(frozen=True, slots=True)
class AllocInput:
    family: str
    vol: float  # per-bar volatility of (recent) returns
    mean: float  # per-bar mean of (recent) returns
    se: float  # standard error of the mean estimate
    is_gp: bool = False


@dataclass(frozen=True, slots=True)
class AllocationConfig:
    gross: float = 1.0
    max_weight: float = 0.34  # per-strategy cap (fraction of gross)
    gp_family_cap: float = 0.2  # max total weight to all gp-* strategies
    min_confidence: float = 0.5  # below this P(mean>0) a champion gets no weight


def _confidence(mean: float, se: float) -> float:
    if se <= 0:
        return 1.0 if mean > 0 else 0.0
    return float(_N.cdf(mean / se))


def allocate(
    inputs: list[AllocInput], corr: np.ndarray | None, cfg: AllocationConfig | None = None
) -> dict[str, float]:
    """Return family -> weight. ``corr`` is the (n x n) return correlation matrix (or None)."""
    cfg = cfg or AllocationConfig()
    n = len(inputs)
    if n == 0:
        return {}

    vols = np.array([max(x.vol, 1e-9) for x in inputs])
    median_vol = float(np.median(vols))
    vols = np.where(vols > 0, vols, median_vol)

    confidence = np.array([_confidence(x.mean, x.se) for x in inputs])
    confidence = np.where(confidence >= cfg.min_confidence, confidence, 0.0)

    if corr is not None and corr.shape == (n, n):
        positive_corr = np.clip(corr, 0.0, None)
        np.fill_diagonal(positive_corr, 0.0)
        corr_penalty = 1.0 / (1.0 + positive_corr.sum(axis=1))
    else:
        corr_penalty = np.ones(n)

    raw = (1.0 / vols) * confidence * corr_penalty
    if raw.sum() <= 0:
        return {x.family: 0.0 for x in inputs}

    weights = raw / raw.sum() * cfg.gross
    weights = _apply_caps(weights, inputs, cfg)
    return {x.family: float(w) for x, w in zip(inputs, weights, strict=True)}


def _apply_caps(weights: np.ndarray, inputs: list[AllocInput], cfg: AllocationConfig) -> np.ndarray:
    weights = np.minimum(weights, cfg.max_weight * cfg.gross)

    gp_mask = np.array([x.is_gp for x in inputs])
    gp_total = float(weights[gp_mask].sum()) if gp_mask.any() else 0.0
    cap = cfg.gp_family_cap * cfg.gross
    if gp_total > cap and gp_total > 0:
        weights[gp_mask] *= cap / gp_total

    total = float(weights.sum())
    if total > cfg.gross and total > 0:
        weights *= cfg.gross / total  # never forced up; a cash buffer is acceptable
    return weights
