"""The vectorized backtest core."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.backtest.metrics import compute_stats


@dataclass(slots=True)
class BacktestResult:
    equity: pd.Series  # cumulative equity, starts at 1.0
    returns: pd.Series  # per-bar net returns
    applied_weights: pd.Series  # weights actually in force each bar (shifted by 1)
    turnover: pd.Series
    stats: dict[str, float]


def run_backtest(
    close: pd.Series,
    weights: pd.Series,
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    periods_per_year: float = 365.0,
) -> BacktestResult:
    """Backtest a target-weight series against a price series.

    No look-ahead: the weight decided on bar ``t`` is shifted and only earns bar ``t+1``'s
    return. Costs are charged on turnover (the absolute change in applied weight) at
    ``fee_bps + slippage_bps`` per unit of traded weight.
    """
    close = close.astype(float)
    weights = weights.reindex(close.index).fillna(0.0).clip(-1.0, 1.0)

    asset_ret = close.pct_change().fillna(0.0)
    applied = weights.shift(1).fillna(0.0)
    gross = applied * asset_ret

    turnover = (applied - applied.shift(1)).abs().fillna(applied.abs())
    cost = turnover * (fee_bps + slippage_bps) / 10_000.0
    net = gross - cost

    equity = (1.0 + net).cumprod()
    stats = compute_stats(net, equity, applied, turnover, periods_per_year)
    return BacktestResult(
        equity=equity, returns=net, applied_weights=applied, turnover=turnover, stats=stats
    )


def backtest_strategy(
    strategy,
    bars: pd.DataFrame,
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    periods_per_year: float = 365.0,
) -> BacktestResult:
    """Backtest one strategy instance on ``bars``.

    A strategy whose fills cannot be expressed as close-to-close weights (resting orders
    that fill inside the bar) provides its own ``backtest``; every other strategy is
    run through :func:`run_backtest` on its weights.
    """
    own = getattr(strategy, "backtest", None)
    if callable(own):
        return own(
            bars,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
        )
    return run_backtest(
        bars["close"],
        strategy.weights(bars),
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        periods_per_year=periods_per_year,
    )


def run_backtest_panel(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    periods_per_year: float = 365.0,
) -> BacktestResult:
    """Backtest a multi-asset target-weight panel against a close panel.

    Same no-look-ahead rule as :func:`run_backtest`: each row of weights is shifted one bar
    before being applied. Portfolio return is the sum across assets; turnover is the summed
    absolute change in applied weight across assets. ``applied_weights`` in the result is
    the per-bar **gross** exposure (sum of absolute applied weights).
    """
    closes = closes.astype(float)
    weights = weights.reindex(index=closes.index, columns=closes.columns).fillna(0.0)

    asset_ret = closes.pct_change().fillna(0.0)
    applied = weights.shift(1).fillna(0.0)
    gross = (applied * asset_ret).sum(axis=1)

    delta = (applied - applied.shift(1)).abs()
    turnover = delta.sum(axis=1)
    turnover.iloc[0] = applied.iloc[0].abs().sum()
    cost = turnover * (fee_bps + slippage_bps) / 10_000.0
    net = gross - cost

    equity = (1.0 + net).cumprod()
    gross_exposure = applied.abs().sum(axis=1)
    stats = compute_stats(net, equity, gross_exposure, turnover, periods_per_year)
    return BacktestResult(
        equity=equity,
        returns=net,
        applied_weights=gross_exposure,
        turnover=turnover,
        stats=stats,
    )
