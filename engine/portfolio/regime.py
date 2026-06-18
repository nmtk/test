"""Regime-aware exposure scaling (volatility targeting).

Rather than a fragile regime *classifier*, we use a robust, well-grounded control: scale an
asset's exposure down when its recent volatility runs hot relative to its long-run norm
(turbulent / risk-off regimes), and never scale above 1 (no leverage from this control).
This damps drawdowns in exactly the regimes where strategies tend to break.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    vol_window: int = 20  # "recent" volatility window
    long_window: int = 120  # long-run volatility (the target)
    floor: float = 0.25  # never scale exposure below this fraction


def exposure_scale(close: pd.Series, cfg: RegimeConfig | None = None) -> float:
    """Return an exposure multiplier in [floor, 1] for an asset from its close series."""
    cfg = cfg or RegimeConfig()
    returns = close.astype(float).pct_change().dropna()
    if len(returns) < cfg.long_window:
        return 1.0
    recent = float(returns.tail(cfg.vol_window).std(ddof=0))
    target = float(returns.tail(cfg.long_window).std(ddof=0))
    if recent <= 0 or target <= 0:
        return 1.0
    return float(min(1.0, max(cfg.floor, target / recent)))


def regime_label(close: pd.Series, cfg: RegimeConfig | None = None) -> str:
    """Coarse human-readable regime tag for reports (not used for sizing)."""
    cfg = cfg or RegimeConfig()
    scale = exposure_scale(close, cfg)
    vol_state = "turbulent" if scale < 1.0 else "calm"
    if len(close) > cfg.long_window:
        trend = "up" if close.iloc[-1] >= close.tail(cfg.long_window).mean() else "down"
    else:
        trend = "flat"
    return f"{vol_state}/{trend}"
