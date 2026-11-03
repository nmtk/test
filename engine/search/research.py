"""The nightly research pipeline: search every archetype/symbol/timeframe, validate, incubate.

For each (archetype, symbol, timeframe) it runs optuna search + the validation gate and
records the outcome in the population. A strategy that *passes* the gate enters
``incubating`` (it is NOT allocated yet) — it must first survive shadow incubation on
forward data, handled by the promotion FSM (``engine.orchestration.promotion``). This split
is deliberate: the search/generator never promotes straight to live capital.

Families already held by an incumbent — incubating or champion — are skipped, not
re-searched. A row under forward test carries state the search must not touch, above all
the clock that test is measured from (see ``held_incumbent``).
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field

from engine.backtest import periods_per_year
from engine.core.config import Settings, load_universe
from engine.core.types import AssetClass
from engine.data.intrabar import IntrabarStore, prepare_intrabar_bars
from engine.data.store import OHLCVStore
from engine.execution.broker import intrabar_research_costs, research_costs
from engine.features.context import ContextAugmenter
from engine.gp.evolve import GPConfig
from engine.orchestration.population import Population, StrategyRecord
from engine.search.ensemble import discover_ensemble
from engine.search.gp_search import discover_gp
from engine.search.optuna_search import discover_challenger
from engine.strategies import get_strategy
from engine.validation.gate import GateThresholds
from engine.validation.ledger import TrialLedger

DEFAULT_ARCHETYPES = (
    "tsmom",
    "ma_cross",
    "donchian",
    "zscore_reversion",
    "macro_flow",
    "wemof",
    "wemof_tick",
    "trade_balance",
    "carry",
    "session",
    "vol_regime",
)
MIN_BARS = 300

# GP is the most overfitting-prone generator (see docs/MANUAL.md §5), so it faces a
# deliberately stricter gate than the parameter-search archetypes.
GP_GATE = GateThresholds(
    dsr_min=0.99, pbo_max=0.10, spa_max=0.02, min_oos_sharpe=0.0, min_pct_positive=0.55
)

# A family in one of these states belongs to an *incumbent* and research leaves it alone.
HELD_STATUSES = frozenset({"incubating", "champion"})


@dataclass(slots=True)
class ResearchSummary:
    n_evaluated: int = 0
    n_passed: int = 0
    n_held: int = 0
    families: list[str] = field(default_factory=list)


def held_incumbent(existing: dict | None) -> bool:
    """True when this family's row belongs to an incumbent research must not overwrite.

    An ``incubating`` row is a *test in progress*. Two things in it are state, not output:
    ``incubation_start_ts`` is the clock the promotion FSM counts forward bars from, and
    its stored metrics are the reference band that forward performance is judged against.
    A ``champion`` row is a test that passed and now holds capital.

    Without this guard the nightly run re-searched every family and upserted whatever it
    found, which reset ``incubation_start_ts`` to the newest bar *every night*. The
    forward window therefore never grew past a bar or two and
    ``PromotionConfig.min_incubation_bars`` (60) was unreachable in principle: five months
    of nightly runs produced repeated gate passes and never one champion. Snapshots of
    ``state/population.db`` show the clock stepping forward a day at a time (2026-10-26,
    then 10-28, then 10-31, then 11-02) while the forward window stayed at one bar.

    Only the promotion FSM may move an incumbent: promote it, retire it on decay, or
    reject it when its incubation window closes without the evidence holding up. Research
    takes the family back over on the night after that verdict.

    A research-only row is excluded, because the FSM deliberately never touches one (no
    venue can fill it, see ``engine.orchestration.promotion``). Holding one would freeze
    its family for good rather than for its incubation window.
    """
    if not existing or existing.get("status") not in HELD_STATUSES:
        return False
    return not (existing.get("metrics") or {}).get("research_only")


def _metrics(report, ppy: float, fee_bps: float = 10.0, slippage_bps: float = 5.0) -> dict:
    """Validation outcome + backtest reference stats the incubation/decay bands compare to.

    The cost assumption is recorded alongside the stats so the promotion FSM judges forward
    performance against the *same* costs the gate validated under. Without that, an FX
    champion validated at 1bps would be re-measured at crypto's 15bps and look decayed.
    """
    return {
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
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


def _record_for(
    report,
    *,
    family,
    strat_name,
    symbol,
    source,
    tf,
    ppy,
    last_ts,
    existing,
    fee_bps=10.0,
    slippage_bps=5.0,
    research_only=False,
):
    """Decide the population row for a discovery, preserving an incumbent's row and clocks."""
    # An incumbent is never overwritten by a fresh discovery (see held_incumbent). Handing
    # back its row verbatim keeps its incubation clock and its reference metrics intact.
    # run_research skips the search for these families; this is the backstop for callers
    # that reach _record_for another way (e.g. `research --strategies X` on one family).
    if held_incumbent(existing):
        return StrategyRecord(
            family=family,
            strategy=existing["strategy"],
            symbol=existing["symbol"],
            source=existing["source"],
            timeframe=existing["timeframe"],
            params=existing["params"],
            status=existing["status"],
            metrics=existing["metrics"],
            incubation_start_ts=existing.get("incubation_start_ts"),
            champion_start_ts=existing.get("champion_start_ts"),
        )

    metrics = _metrics(report, ppy, fee_bps=fee_bps, slippage_bps=slippage_bps)
    metrics["research_only"] = bool(research_only)
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

    # The family is free (candidate, retired, or new), so tonight's pass takes it and the
    # forward clock starts at the last bar the gate saw. Incumbents never get here —
    # held_incumbent returned their row above, which is also what keeps a champion's
    # reference metrics the ones it was validated against rather than a nightly re-fit.
    return StrategyRecord(**base, status="incubating", incubation_start_ts=last_ts)


def run_research(
    settings: Settings,
    *,
    strategies: tuple[str, ...] | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    trials: int = 40,
    thresholds: GateThresholds | None = None,
    # None = use the per-asset-class research assumption (see broker.RESEARCH_COSTS_BPS).
    # Charging crypto's fees to an FX strategy would reject real edges out of hand.
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
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
    intrabar_store = IntrabarStore(settings.data_dir / "intrabar")
    population = Population(settings.state_dir / "population.db")
    ledger = TrialLedger.load(settings.state_dir / "trial_ledger.json")
    thresholds = thresholds or GateThresholds()
    archetypes = strategies or DEFAULT_ARCHETYPES

    summary = ResearchSummary()
    try:
        # Crypto and spot FX. Equities are excluded because Alpaca's free IEX feed only
        # backfills from the day keys are issued, so there is rarely enough history to
        # clear the gate; FX has three years of free tick history from bar one.
        researchable = [
            *universe.by_asset_class(AssetClass.CRYPTO),
            *universe.by_asset_class(AssetClass.FX),
        ]
        # Research-only series carry deep history for instruments whose tradeable feed
        # is shallow. They are researched exactly like tradeable ones and then barred
        # from promotion (see engine/orchestration/promotion.py), because their source is
        # not a venue any order can reach.
        Researchable = namedtuple("Researchable", "symbol source asset_class timeframes only")
        targets = [
            Researchable(
                e.instrument.symbol,
                e.live_source.source,
                e.instrument.asset_class,
                e.timeframes,
                False,
            )
            for e in researchable
        ] + [
            Researchable(r.symbol, r.source, r.asset_class, r.timeframes, True)
            for r in universe.research_series
        ]
        for entry in targets:
            symbol = entry.symbol
            if symbols and symbol not in symbols:
                continue
            source = entry.source
            asset_class = entry.asset_class
            entry_fee, entry_slip = research_costs(asset_class)
            if fee_bps is not None:
                entry_fee = fee_bps
            if slippage_bps is not None:
                entry_slip = slippage_bps
            for tf in entry.timeframes:
                if timeframes and tf not in timeframes:
                    continue
                bars = store.read(source, symbol, tf)
                if len(bars) < min_bars:
                    continue
                bars = augmenter.augment(bars, source=source, symbol=symbol, timeframe=tf)
                ppy = periods_per_year(tf, asset_class)
                last_ts = int(bars["ts"].iloc[-1])
                for strat_name in archetypes:
                    family = f"{strat_name}:{symbol}:{tf}"
                    existing = population.get(family)
                    if held_incumbent(existing):
                        # Under forward test or holding capital: skipped before the search
                        # so the night spends neither compute nor trial-ledger budget on a
                        # family whose row it is not allowed to change.
                        summary.n_held += 1
                        continue
                    strategy_cls = get_strategy(strat_name)
                    run_bars, run_fee, run_slip = bars, entry_fee, entry_slip
                    run_only, inherited = entry.only, 0
                    if getattr(strategy_cls, "requires_intrabar", False):
                        # Skipped *before* the search, so nights without a fully synced
                        # path spend nothing from the trial ledger.
                        if tf not in strategy_cls.intrabar_timeframes:
                            continue
                        run_bars, _ = prepare_intrabar_bars(
                            bars, intrabar_store, source=source, symbol=symbol, timeframe=tf
                        )
                        if run_bars is None or len(run_bars) < min_bars:
                            continue
                        run_fee, run_slip = intrabar_research_costs(asset_class)
                        if fee_bps is not None:
                            run_fee = fee_bps
                        if slippage_bps is not None:
                            run_slip = slippage_bps
                        # No executor can place its intrabar brackets (see wemof_tick.py).
                        run_only = True
                        inherited = ledger.get(f"{strategy_cls.parent_family}:{symbol}:{tf}")
                    report = discover_challenger(
                        run_bars,
                        strategy_cls,
                        n_trials=trials,
                        ppy=ppy,
                        fee_bps=run_fee,
                        slippage_bps=run_slip,
                        seed=seed,
                        thresholds=thresholds,
                        ledger=ledger,
                        family=family,
                        inherited_trials=inherited,
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
                            last_ts=int(run_bars["ts"].iloc[-1]),
                            existing=existing,
                            fee_bps=run_fee,
                            slippage_bps=run_slip,
                            research_only=run_only,
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
                    existing = population.get(family)
                    if held_incumbent(existing):
                        # Deliberate side effect: the hall of fame for this symbol/timeframe
                        # stops accumulating while the gp family is held, because evolving
                        # it would also spend ledger budget on a row we may not write.
                        summary.n_held += 1
                    else:
                        report = discover_gp(
                            bars,
                            ppy=ppy,
                            cfg=gp_config,
                            seed=seed,
                            fee_bps=entry_fee,
                            slippage_bps=entry_slip,
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
                                existing=existing,
                                fee_bps=entry_fee,
                                slippage_bps=entry_slip,
                                research_only=entry.only,
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
                    existing = population.get(family)
                    if held_incumbent(existing):
                        summary.n_held += 1
                        continue
                    report = discover_ensemble(
                        bars,
                        population=population,
                        symbol=symbol,
                        timeframe=tf,
                        archetypes=archetypes,
                        hof_path=hof_path,
                        ppy=ppy,
                        fee_bps=entry_fee,
                        slippage_bps=entry_slip,
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
                                existing=existing,
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
