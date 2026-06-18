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
from engine.portfolio.allocate import AllocationConfig, AllocInput, allocate
from engine.portfolio.regime import RegimeConfig, exposure_scale
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


# How many recent bars of forward returns to estimate allocation stats / correlations from.
RECENT_ALLOC_BARS = 90


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


def _strategy_returns(store: OHLCVStore, record: dict, cfg: PromotionConfig):
    """Full net-return series for a champion (used to estimate allocation stats)."""
    bars = store.read(record["source"], record["symbol"], record["timeframe"])
    if len(bars) < 2:
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
    return result.returns


def allocate_champions(
    population: Population,
    store: OHLCVStore,
    *,
    promo_cfg: PromotionConfig,
    alloc_cfg: AllocationConfig | None = None,
    regime_cfg: RegimeConfig | None = None,
    recent_bars: int = RECENT_ALLOC_BARS,
) -> Champions:
    """Confidence-weighted, correlation-aware, regime-scaled allocation over champions."""
    alloc_cfg = alloc_cfg or AllocationConfig()
    regime_cfg = regime_cfg or RegimeConfig()

    champs = population.list("champion")
    champs.sort(
        key=lambda r: (r["metrics"].get("dsr", 0.0), r["metrics"].get("cpcv_median", 0.0)),
        reverse=True,
    )
    champs = champs[: promo_cfg.max_champions]

    # Recent forward returns per champion -> allocation stats + correlation matrix.
    recent: dict[str, np.ndarray] = {}
    for rec in champs:
        series = _strategy_returns(store, rec, promo_cfg)
        if series is not None and len(series) >= 10:
            recent[rec["family"]] = np.asarray(series.tail(recent_bars), dtype=float)
    champs = [c for c in champs if c["family"] in recent]
    if not champs:
        return Champions(allocations=[], updated=datetime.now(UTC).isoformat())

    length = min(len(recent[c["family"]]) for c in champs)
    matrix = np.vstack([recent[c["family"]][-length:] for c in champs])
    inputs = [
        AllocInput(
            family=c["family"],
            vol=float(np.std(row, ddof=1)) if length > 1 else 0.0,
            mean=float(np.mean(row)),
            se=(float(np.std(row, ddof=1)) / np.sqrt(length)) if length > 1 else 0.0,
            is_gp=(c["strategy"] == "gp"),
        )
        for c, row in zip(champs, matrix, strict=True)
    ]
    corr = (
        np.nan_to_num(np.corrcoef(matrix), nan=0.0) if (len(champs) >= 2 and length >= 3) else None
    )
    weights = allocate(inputs, corr, alloc_cfg)

    allocations = []
    for c in champs:
        bars = store.read(c["source"], c["symbol"], c["timeframe"])
        scale = exposure_scale(bars["close"], regime_cfg) if len(bars) else 1.0
        allocations.append(
            Allocation(
                strategy=c["strategy"],
                symbol=c["symbol"],
                source=c["source"],
                timeframe=c["timeframe"],
                weight=round(weights.get(c["family"], 0.0) * scale, 4),
                params=c["params"],
            )
        )
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

        champions = allocate_champions(population, store, promo_cfg=cfg)
        save_champions(settings.state_dir / "champions.json", champions)
        summary.n_champions = len(champions.allocations)
    finally:
        population.close()
    return summary
