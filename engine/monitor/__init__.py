"""Decay / regime-break detection on a strategy's realized return stream.

Self-implemented Page-Hinkley and one-sided CUSUM (no ``river`` dependency) plus a
Sharpe confidence band. These are the statistical tripwires that demote a champion whose
live (paper) behavior drifts below what its backtest promised — the core of capital
protection.
"""

from engine.monitor.decay import (
    cusum_down,
    page_hinkley,
    sharpe_lower_band,
)

__all__ = ["page_hinkley", "cusum_down", "sharpe_lower_band"]
