"""Performance metrics and annualization helpers."""

from __future__ import annotations

import math

import pandas as pd

from engine.core.clock import timeframe_seconds
from engine.core.types import AssetClass

_SECONDS_PER_DAY = 86_400
_EQUITY_SESSION_SECONDS = 6.5 * 3600  # NYSE regular session
_EQUITY_SESSIONS_PER_YEAR = 252


def periods_per_year(timeframe: str, asset_class: AssetClass) -> float:
    """Number of bars per year, used to annualize Sharpe/vol/return.

    Crypto trades continuously; equities only during ~252 sessions of ~6.5h.
    """
    sec = timeframe_seconds(timeframe)
    if asset_class is AssetClass.CRYPTO:
        return 365 * _SECONDS_PER_DAY / sec
    if sec >= _SECONDS_PER_DAY:
        return _EQUITY_SESSIONS_PER_YEAR * _SECONDS_PER_DAY / sec
    return _EQUITY_SESSIONS_PER_YEAR * _EQUITY_SESSION_SECONDS / sec


def sharpe_ratio(returns: pd.Series, ppy: float) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=0))
    if sd == 0.0:
        return 0.0
    return float(r.mean() / sd * math.sqrt(ppy))


def annual_volatility(returns: pd.Series, ppy: float) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=0) * math.sqrt(ppy))


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough fraction (e.g. -0.25 for a 25% drawdown)."""
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def cagr(equity: pd.Series, ppy: float) -> float:
    if len(equity) < 2:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0])
    n = len(equity) - 1
    if total <= 0.0:
        return -1.0
    return float(total ** (ppy / n) - 1.0)


def compute_stats(
    returns: pd.Series,
    equity: pd.Series,
    applied_weights: pd.Series,
    turnover: pd.Series,
    ppy: float,
) -> dict[str, float]:
    active = returns[applied_weights != 0.0]
    wins = int((active > 0).sum())
    n_active = int(len(active))
    return {
        "sharpe": sharpe_ratio(returns, ppy),
        "cagr": cagr(equity, ppy),
        "ann_vol": annual_volatility(returns, ppy),
        "max_drawdown": max_drawdown(equity),
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "n_bars": int(len(returns)),
        "turnover": float(turnover.sum()),
        "avg_exposure": float(applied_weights.abs().mean()) if len(applied_weights) else 0.0,
        "hit_rate": float(wins / n_active) if n_active else 0.0,
    }
