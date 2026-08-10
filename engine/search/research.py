"""The nightly research pipeline: search every archetype/symbol/timeframe, validate, incubate.

For each (archetype, symbol, timeframe) it runs optuna search + the validation gate and
records the outcome in the population. A strategy that *passes* the gate enters
``incubating`` (it is NOT allocated yet) — it must first survive shadow incubation on
forward data, handled by the promotion FSM (``engine.orchestration.promotion``). This split
is deliberate: the search/generator never promotes straight to live capital.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.backtest import periods_per_year
from engine.core.config import Settings, load_universe
from engine.core.types import AssetClass
from engine.data.store import OHLCVStore
from engine.features.context import ContextAugmenter
from engine.gp.evolve import GPConfig
from engine.orchestration.population import Population, StrategyRecord
from engine.search.ensemble import discover_ensemble
from engine.search.gp_search import discover_gp
from engine.search.optuna_search import discover_challenger
from engine.strategies import get_strategy
from engine.validation.gate import GateThresholds
from engine.validation.ledger import TrialLedger

DEFAULT_ARCHETYPES = ("tsmom", "ma_cross", "donchian", "zscore_reversion", "macro_flow", "wemof")
MIN_BARS = 300

# GP is the most overfitting-prone generator (see docs/MANUAL.md §5), so it faces a
# deliberately stricter gate than the parameter-search archetypes.
GP_GATE = GateThresholds(
    dsr_min=0.99, pbo_max=0.10, spa_max=0.02, min_oos_sharpe=0.0, min_pct_positive=0.55
)


@dataclass(slots=True)
class ResearchSummary:
    n_evaluated: int = 0
    n_passed: int = 0
    families: list[str] = field(default_factory=list)


def _metrics(report, ppy: float) -> dict:
    """Validation outcome + backtest reference stats the incubation/decay bands compare to."""
    return {
        "dsr": report.dsr,
        "pbo": report.pbo,
        "spa_p": report.spa_p,
        "cpcv_median": report.cpcv_median,
        "sharpe_annual": report.sharpe_annual,
        "passed": report.passed,
        # reference (per-bar) backtest stats for forward-test bands:
        "sharpe_per_bar": report.sharpe_per_bar,
        "mean_per_bar": report.mean_per_bar,
        "std_per_bar": report.std_per_bar,
        "skew": report.skew,
        "kurtosis": report.kurtosis,
        "max_drawdown": report.max_drawdown,
        "n": report.n_bars,
        "ppy": ppy,
    }


def _record_for(report, *, family, strat_name, symbol, source, tf, ppy, last_ts, existing):
    """Decide the population row for a discovery, preserving a still-valid champion."""
    metrics = _metrics(report, ppy)
    base = dict(
        family=family,
        strategy=strat_name,
        symbol=symbol,
        source=source,
        timeframe=tf,
        params=report.best_params,
        metrics=metrics,
    )
    if not report.passed:
        return StrategyRecord(**base, status="candidate")

    # A still-valid champion with unchanged params keeps its status and clocks.
    if existing and existing["status"] == "champion" and existing["params"] == report.best_params:
        return StrategyRecord(
            **base,
            status="champion",
            incubation_start_ts=existing.get("incubation_start_ts"),
            champion_start_ts=existing.get("champion_start_ts"),
        )
    # New / changed / previously-failed -> (re)start incubation on forward data.
    return StrategyRecord(**base, status="incubating", incubation_start_ts=last_ts)


def run_research(
    settings: Settings,
    *,
    strategies: tuple[str, ...] | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    trials: int = 40,
    thresholds: GateThresholds | None = None,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    seed: int = 0,
    min_bars: int = MIN_BARS,
    include_gp: bool = False,
    gp_config: GPConfig | None = None,
    include_ensemble: bool = True,
) -> ResearchSummary:
    universe = load_universe(settings.universe_path)
    store = OHLCVStore(settings.data_dir / "ohlcv")
    # Cross-asset context (macro + BTC market factor) is stamped onto every bar frame so
    # both the archetype search and GP evolve over flow-conditional features. The paper
    # and live executors apply the identical augmentation (see orchestrator/live).
    augmenter = ContextAugmenter.from_settings(settings, universe)
    population = Population(settings.state_dir / "population.db")
    ledger = TrialLedger.load(settings.state_dir / "trial_ledger.json")
    thresholds = thresholds or GateThresholds()
    archetypes = strategies or DEFAULT_ARCHETYPES

    summary = ResearchSummary()
    try:
        for entry in universe.by_asset_class(AssetClass.CRYPTO):
            symbol = entry.instrument.symbol
            if symbols and symbol not in symbols:
                continue
            source = entry.live_source.source
            for tf in entry.timeframes:
                if timeframes and tf not in timeframes:
                    continue
                bars = store.read(source, symbol, tf)
                if len(bars) < min_bars:
                    continue
                bars = augmenter.augment(bars, source=source, symbol=symbol, timeframe=tf)
                ppy = periods_per_year(tf, entry.instrument.asset_class)
                last_ts = int(bars["ts"].iloc[-1])
                for strat_name in archetypes:
                    family = f"{strat_name}:{symbol}:{tf}"
                    report = discover_challenger(
                        bars,
                        get_strategy(strat_name),
                        n_trials=trials,
                        ppy=ppy,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                        seed=seed,
                        thresholds=thresholds,
                        ledger=ledger,
                        family=family,
                    )
                    population.upsert(
                        _record_for(
                            report,
                            family=family,
                            strat_name=strat_name,
                            symbol=symbol,
                            source=source,
                            tf=tf,
                            ppy=ppy,
                            last_ts=last_ts,
                            existing=population.get(family),
                        )
                    )
                    summary.n_evaluated += 1
                    summary.families.append(family)
                    if report.passed:
                        summary.n_passed += 1

                # HOF is persisted per family so each nightly run warm-starts from the
                # prior run's survivors, accumulating GP knowledge over time. The alpha
                # pool below also draws its GP members from here.
                hof_path = settings.state_dir / "gp_hof" / f"{symbol.replace('/', '-')}_{tf}.json"

                if include_gp:
                    family = f"gp:{symbol}:{tf}"
                    report = discover_gp(
                        bars,
                        ppy=ppy,
                        cfg=gp_config,
                        seed=seed,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                        thresholds=GP_GATE,  # stricter than the archetype gate
                        ledger=ledger,
                        family=family,
                        hof_path=hof_path,
                    )
                    population.upsert(
                        _record_for(
                            report,
                            family=family,
                            strat_name="gp",
                            symbol=symbol,
                            source=source,
                            tf=tf,
                            ppy=ppy,
                            last_ts=last_ts,
                            existing=population.get(family),
                        )
                    )
                    summary.n_evaluated += 1
                    summary.families.append(family)
                    if report.passed:
                        summary.n_passed += 1

                if include_ensemble:
                    # The alpha pool: combine tonight's best archetype configs + the GP
                    # hall of fame into fixed-weight pool variants and gate the family.
                    # Individually-rejected members are allowed in — the pool is judged
                    # as its own family, with the members' trial counts inherited into
                    # its DSR deflation (see engine/search/ensemble.py).
                    family = f"ensemble:{symbol}:{tf}"
                    report = discover_ensemble(
                        bars,
                        population=population,
                        symbol=symbol,
                        timeframe=tf,
                        archetypes=archetypes,
                        hof_path=hof_path,
                        ppy=ppy,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                        thresholds=thresholds,
                        ledger=ledger,
                        family=family,
                    )
                    if report is not None:
                        population.upsert(
                            _record_for(
                                report,
                                family=family,
                                strat_name="ensemble",
                                symbol=symbol,
                                source=source,
                                tf=tf,
                                ppy=ppy,
                                last_ts=last_ts,
                                existing=population.get(family),
                            )
                        )
                        summary.n_evaluated += 1
                        summary.families.append(family)
                        if report.passed:
                            summary.n_passed += 1
        ledger.save()
    finally:
        population.close()
    return summary
