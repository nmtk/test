"""Promotion & protection FSM: shadow-incubate, promote, and retire on decay.

This is the guardrail that must exist *before* a prolific generator (GP, M6) is connected.
A strategy that clears the backtest gate enters ``incubating`` and is **shadow-tested on
genuinely forward data** (bars that arrived after discovery — never seen by the search).
Only if its forward performance stays within the backtest's Sharpe confidence band for long
enough is it promoted to ``champion`` and allocated. Champions are continuously watched; if
forward behavior decays (Page-Hinkley downshift, Sharpe below band, or drawdown beyond a
multiple of backtest), they are retired and de-allocated.

Forward returns are *recomputed deterministically from stored bars* (shadow), so no
per-strategy executed sub-account is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

import engine.gp.strategy  # noqa: F401  -- registers the "gp" strategy for get_strategy
from engine.backtest import run_backtest
from engine.backtest.metrics import max_drawdown
from engine.core.config import Settings
from engine.data.store import OHLCVStore
from engine.monitor.decay import page_hinkley, sharpe_lower_band
from engine.orchestration.champions import Allocation, Champions, save_champions
from engine.orchestration.population import Population
from engine.strategies import get_strategy
from engine.validation.stats import return_moments


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    min_incubation_bars: int = 60  # forward bars required before promotion
    min_decay_bars: int = 30  # forward bars required before judging decay
    band_z: float = 1.645  # one-sided 95% lower band on backtest Sharpe
    max_dd_multiple: float = 1.5  # forward DD beyond this x backtest DD => decay
    ph_threshold: float = 8.0  # Page-Hinkley trigger on standardized forward returns
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    max_champions: int = 8
    gross: float = 1.0


@dataclass(slots=True)
class PromotionSummary:
    n_incubating: int = 0
    n_promoted: int = 0
    n_champions_checked: int = 0
    n_retired: int = 0
    n_champions: int = 0


def forward_returns(store: OHLCVStore, record: dict, cfg: PromotionConfig, start_ts: int):
    """Recompute the strategy's net returns on bars strictly after ``start_ts`` (forward/OOS)."""
    bars = store.read(record["source"], record["symbol"], record["timeframe"])
    if len(bars) == 0 or start_ts is None:
        return None
    ppy = float(record["metrics"].get("ppy", 365.0))
    weights = get_strategy(record["strategy"])(**record["params"]).weights(bars)
    result = run_backtest(
        bars["close"],
        weights,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        periods_per_year=ppy,
    )
    mask = (bars["ts"] > start_ts).to_numpy()
    return result.returns[mask]


def incubation_passed(forward, ref: dict, cfg: PromotionConfig) -> bool:
    """Forward Sharpe must be positive and within the backtest's lower confidence band."""
    if forward is None or len(forward) < cfg.min_incubation_bars:
        return False
    fwd = return_moments(forward)
    lower = sharpe_lower_band(
        ref.get("sharpe_per_bar", 0.0),
        int(ref.get("n", 0)),
        ref.get("skew", 0.0),
        ref.get("kurtosis", 3.0),
        z=cfg.band_z,
    )
    return fwd.sharpe > 0.0 and fwd.sharpe >= lower


def decay_detected(forward, ref: dict, cfg: PromotionConfig) -> list[str]:
    """Return the list of decay reasons (empty == healthy)."""
    if forward is None or len(forward) < cfg.min_decay_bars:
        return []
    reasons: list[str] = []
    fwd = return_moments(forward)

    lower = sharpe_lower_band(
        ref.get("sharpe_per_bar", 0.0),
        int(ref.get("n", 0)),
        ref.get("skew", 0.0),
        ref.get("kurtosis", 3.0),
        z=cfg.band_z,
    )
    if fwd.sharpe < lower:
        reasons.append("below_sharpe_band")

    ref_std = ref.get("std_per_bar", 0.0)
    ref_mean = ref.get("mean_per_bar", 0.0)
    if ref_std > 0:
        standardized = (np.asarray(forward, dtype=float) - ref_mean) / ref_std
        if page_hinkley(standardized, threshold=cfg.ph_threshold):
            reasons.append("page_hinkley")

    ref_dd = ref.get("max_drawdown", 0.0)
    fwd_dd = max_drawdown((1.0 + forward).cumprod())
    if ref_dd < 0 and fwd_dd < cfg.max_dd_multiple * ref_dd:
        reasons.append("drawdown")
    return reasons


def allocate_champions(
    population: Population, *, max_champions: int = 8, gross: float = 1.0
) -> Champions:
    """Build champions.json from champion-status rows (top-N by DSR, equal weight)."""
    champs = population.list("champion")
    champs.sort(
        key=lambda r: (r["metrics"].get("dsr", 0.0), r["metrics"].get("cpcv_median", 0.0)),
        reverse=True,
    )
    selected = champs[:max_champions]
    weight = round(gross / len(selected), 4) if selected else 0.0
    allocations = [
        Allocation(
            strategy=r["strategy"],
            symbol=r["symbol"],
            source=r["source"],
            timeframe=r["timeframe"],
            weight=weight,
            params=r["params"],
        )
        for r in selected
    ]
    return Champions(allocations=allocations, updated=datetime.now(UTC).isoformat())


def _last_ts(store: OHLCVStore, record: dict) -> int | None:
    return store.last_ts(record["source"], record["symbol"], record["timeframe"])


def promote_step(settings: Settings, *, cfg: PromotionConfig | None = None) -> PromotionSummary:
    """One pass of the FSM: incubating->champion on forward evidence, champion->retired on decay."""
    cfg = cfg or PromotionConfig()
    store = OHLCVStore(settings.data_dir / "ohlcv")
    population = Population(settings.state_dir / "population.db")
    summary = PromotionSummary()
    try:
        for rec in population.list("incubating"):
            summary.n_incubating += 1
            fwd = forward_returns(store, rec, cfg, rec.get("incubation_start_ts"))
            if incubation_passed(fwd, rec["metrics"], cfg):
                population.update_fields(
                    rec["family"], status="champion", champion_start_ts=_last_ts(store, rec)
                )
                summary.n_promoted += 1

        for rec in population.list("champion"):
            summary.n_champions_checked += 1
            fwd = forward_returns(store, rec, cfg, rec.get("champion_start_ts"))
            if decay_detected(fwd, rec["metrics"], cfg):
                population.update_fields(rec["family"], status="retired")
                summary.n_retired += 1

        champions = allocate_champions(population, max_champions=cfg.max_champions, gross=cfg.gross)
        save_champions(settings.state_dir / "champions.json", champions)
        summary.n_champions = len(champions.allocations)
    finally:
        population.close()
    return summary
