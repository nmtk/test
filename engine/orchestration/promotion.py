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

Incubation is time-boxed and ends in a verdict. Once ``min_incubation_bars`` of forward
data exist, the row is either promoted or demoted back to ``candidate`` — it never waits
indefinitely. This FSM is the *only* thing allowed to move an incubating or champion row:
the nightly search holds off on those families entirely (see
``engine.search.research.held_incumbent``), because a re-discovery that overwrote the row
would also reset the very clock this module measures forward bars against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil

import numpy as np

import engine.gp.strategy  # noqa: F401  -- registers the "gp" strategy for get_strategy
from engine.backtest import run_backtest
from engine.backtest.metrics import max_drawdown
from engine.core.clock import timeframe_seconds
from engine.core.config import Settings, load_universe
from engine.data.store import OHLCVStore
from engine.features.context import ContextAugmenter
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
    min_incubation_days: float = 60.0  # ...and this much wall-clock, whichever is longer
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


def required_forward_bars(timeframe: str, cfg: PromotionConfig) -> int:
    """Forward bars needed before a verdict, as a *duration* rather than a bar count.

    ``min_incubation_bars`` alone is a trap on fine timeframes. Sixty bars of 5m data is
    five hours. The standard error of a Sharpe estimated from 144 observations is about
    1/sqrt(144) = 0.083 per bar, so a measured 0.055 per bar is indistinguishable from
    zero — and then annualising it by sqrt(105120) prints "+15.1", which reads like a
    spectacular result and is in fact noise. Five hours cannot settle anything, and the
    first batch of shadow verdicts produced exactly that number before this existed.

    So the requirement is whichever is longer: ``min_incubation_bars`` bars, or
    ``min_incubation_days`` of wall clock. On daily bars the two coincide at 60, which is
    what the bar count always meant; on 1h it becomes 1,440 bars and on 5m 17,280.
    """
    step_s = timeframe_seconds(timeframe)
    by_time = ceil(cfg.min_incubation_days * 86_400.0 / step_s) if step_s > 0 else 0
    return max(int(cfg.min_incubation_bars), int(by_time))


@dataclass(slots=True)
class PromotionSummary:
    n_incubating: int = 0
    n_promoted: int = 0
    n_rejected: int = 0
    n_untradeable: int = 0
    n_champions_checked: int = 0
    n_retired: int = 0
    n_champions: int = 0


def _record_costs(record: dict, cfg: PromotionConfig) -> tuple[float, float]:
    """The cost assumption this strategy was validated under, falling back to ``cfg``.

    Forward performance must be judged against the same costs the gate used, or an FX
    strategy validated at ~1bps gets re-measured at crypto's 15bps and is retired as
    "decayed" when nothing about it changed. Records written before costs were tracked
    simply fall back to the config defaults.
    """
    metrics = record.get("metrics") or {}
    return (
        float(metrics.get("fee_bps", cfg.fee_bps)),
        float(metrics.get("slippage_bps", cfg.slippage_bps)),
    )


def forward_returns(
    store: OHLCVStore,
    record: dict,
    cfg: PromotionConfig,
    start_ts: int,
    augmenter: ContextAugmenter | None,
):
    """Recompute the strategy's net returns on bars strictly after ``start_ts`` (forward/OOS).

    ``augmenter`` stamps the ``ctx_*`` and ``mac_*`` columns, and passing it is not
    optional in practice: research computes weights on augmented bars, and so do the paper
    and live executors, so a forward test on raw bars measures *a different strategy than
    the gate validated*. For anything reading a context feature the difference is total —
    the column is missing, the signal reads neutral, and the strategy returns flat. All
    three ``gp:*`` rows incubating on 2026-11-11 had zero exposure across the entire
    series on raw bars and 14-20% once augmented, so their forward Sharpe was structurally
    0.0, which can never satisfy ``incubation_passed``'s ``fwd.sharpe > 0.0``. They were
    not slow to promote; they were unpromotable. Eight of the other rows differed too.

    The parameter is required rather than defaulted so that a caller must decide. ``None``
    is accepted, and means "this strategy uses no context features" — correct only for a
    pure price archetype.
    """
    bars = store.read(record["source"], record["symbol"], record["timeframe"])
    if len(bars) == 0 or start_ts is None:
        return None
    if augmenter is not None:
        bars = augmenter.augment(
            bars,
            source=record["source"],
            symbol=record["symbol"],
            timeframe=record["timeframe"],
        )
    ppy = float(record["metrics"].get("ppy", 365.0))
    weights = get_strategy(record["strategy"])(**record["params"]).weights(bars)
    fee_bps, slippage_bps = _record_costs(record, cfg)
    result = run_backtest(
        bars["close"],
        weights,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        periods_per_year=ppy,
    )
    mask = (bars["ts"] > start_ts).to_numpy()
    return result.returns[mask]


def incubation_passed(
    forward, ref: dict, cfg: PromotionConfig, timeframe: str | None = None
) -> bool:
    """Forward Sharpe must be positive and within the backtest's lower confidence band.

    ``timeframe`` sets how much forward data counts as enough (see
    :func:`required_forward_bars`). It is optional only so existing callers keep working;
    pass it, or a 5m strategy is judged on five hours.
    """
    needed = cfg.min_incubation_bars if timeframe is None else required_forward_bars(timeframe, cfg)
    if forward is None or len(forward) < needed:
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
    fee_bps, slippage_bps = _record_costs(record, cfg)
    result = run_backtest(
        bars["close"],
        weights,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
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
    universe = load_universe(settings.universe_path)
    # The same augmentation research and both executors apply. Built once: it loads the
    # context series from disk.
    augmenter = ContextAugmenter.from_settings(settings, universe)
    summary = PromotionSummary()
    try:
        tradeable = {e.instrument.symbol for e in universe.entries}
        for rec in population.list("incubating"):
            summary.n_incubating += 1
            if (rec.get("metrics") or {}).get("research_only"):
                # Validated on a series that is not a tradeable venue (see
                # ResearchSeries). Promoting it would hand the executor a symbol no
                # broker can fill — _resolve_instrument would fabricate an instrument and
                # try to trade "DEXJPUS". It informs research; it never trades.
                continue
            if rec["symbol"] not in tradeable:
                # The same refusal, decided from the universe instead of the metadata.
                # research_only is a flag the search writes, and a flag can be wrong: the
                # ensemble branch of run_research omitted it entirely, so
                # ensemble:DEXJPUS:1d sat in `incubating` with research_only=False,
                # eligible for promotion to a symbol no venue carries. That is fixed at
                # the source, but a promotion hands an order path real capital, so it does
                # not rest on one flag being right. The universe is the ground truth for
                # what can be filled.
                summary.n_untradeable += 1
                continue
            fwd = forward_returns(store, rec, cfg, rec.get("incubation_start_ts"), augmenter)
            if incubation_passed(fwd, rec["metrics"], cfg, rec["timeframe"]):
                population.update_fields(
                    rec["family"], status="champion", champion_start_ts=_last_ts(store, rec)
                )
                summary.n_promoted += 1
            elif fwd is not None and len(fwd) >= required_forward_bars(rec["timeframe"], cfg):
                # The window is full and the forward evidence did not hold up. Incubation
                # is a test with a verdict, not an open-ended wait: demote the row so the
                # family is searchable again. Research holds off while a row incubates
                # (see engine.search.research.held_incumbent), so without this verdict a
                # failed incumbent would freeze its family permanently.
                population.update_fields(
                    rec["family"], status="candidate", incubation_start_ts=None
                )
                summary.n_rejected += 1

        for rec in population.list("champion"):
            summary.n_champions_checked += 1
            fwd = forward_returns(store, rec, cfg, rec.get("champion_start_ts"), augmenter)
            if decay_detected(fwd, rec["metrics"], cfg):
                population.update_fields(rec["family"], status="retired")
                summary.n_retired += 1

        champions = allocate_champions(population, store, promo_cfg=cfg)
        save_champions(settings.state_dir / "champions.json", champions)
        summary.n_champions = len(champions.allocations)
    finally:
        population.close()
    return summary
