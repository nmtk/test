"""A small, auditable vectorized backtester and its performance metrics.

It is deliberately dependency-light (numpy/pandas only) and free of look-ahead by
construction: target weights are shifted one bar before being applied to returns, so a
weight decided on bar t's close is only earned from bar t+1 onward. A parity test against
the event-driven PaperBroker guards this property.
"""

from engine.backtest.engine import BacktestResult, run_backtest, run_backtest_panel
from engine.backtest.metrics import compute_stats, periods_per_year

__all__ = [
    "BacktestResult",
    "run_backtest",
    "run_backtest_panel",
    "compute_stats",
    "periods_per_year",
]
