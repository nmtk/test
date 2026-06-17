"""The nightly research pipeline: search every archetype/symbol/timeframe, validate, promote.

For each (archetype, symbol, timeframe) it runs optuna search + the validation gate, records
the outcome in the population (challenger if it passed, else candidate), then rebuilds the
champion set from all validated strategies. This is the loop that lets the system *find and
adopt* better strategies over time instead of running a fixed rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from engine.backtest import periods_per_year
from engine.core.config import Settings, load_universe
from engine.core.types import AssetClass
from engine.data.store import OHLCVStore
from engine.orchestration.champions import Allocation, Champions, save_champions
from engine.orchestration.population import Population, StrategyRecord
from engine.search.optuna_search import discover_challenger
from engine.strategies import get_strategy
from engine.validation.gate import GateThresholds
from engine.validation.ledger import TrialLedger

DEFAULT_ARCHETYPES = ("tsmom", "ma_cross", "donchian", "zscore_reversion")
MIN_BARS = 300


@dataclass(slots=True)
class ResearchSummary:
    n_evaluated: int = 0
    n_passed: int = 0
    n_champions: int = 0
    families: list[str] = field(default_factory=list)


def _metrics(report) -> dict:
    return {
        "dsr": report.dsr,
        "pbo": report.pbo,
        "spa_p": report.spa_p,
        "cpcv_median": report.cpcv_median,
        "sharpe_annual": report.sharpe_annual,
        "passed": report.passed,
    }


def build_champions_from_population(
    population: Population, *, max_champions: int = 8, gross: float = 1.0
) -> Champions:
    """Select the best validated strategies as champions; demote the rest to challenger."""
    eligible = population.list("challenger") + population.list("champion")
    eligible.sort(
        key=lambda r: (r["metrics"].get("dsr", 0.0), r["metrics"].get("cpcv_median", 0.0)),
        reverse=True,
    )
    selected = eligible[:max_champions]

    selected_keys = {(r["family"], r["timeframe"]) for r in selected}
    for r in eligible:
        new_status = "champion" if (r["family"], r["timeframe"]) in selected_keys else "challenger"
        if r["status"] != new_status:
            population.set_status(r["family"], r["params"], new_status)

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
    max_champions: int = 8,
    gross: float = 1.0,
    seed: int = 0,
    min_bars: int = MIN_BARS,
) -> ResearchSummary:
    universe = load_universe(settings.universe_path)
    store = OHLCVStore(settings.data_dir / "ohlcv")
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
                ppy = periods_per_year(tf, entry.instrument.asset_class)
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
                        StrategyRecord(
                            family=family,
                            strategy=strat_name,
                            symbol=symbol,
                            source=source,
                            timeframe=tf,
                            params=report.best_params,
                            status="challenger" if report.passed else "candidate",
                            metrics=_metrics(report),
                        )
                    )
                    summary.n_evaluated += 1
                    summary.families.append(family)
                    if report.passed:
                        summary.n_passed += 1

        ledger.save()
        champions = build_champions_from_population(
            population, max_champions=max_champions, gross=gross
        )
        save_champions(settings.state_dir / "champions.json", champions)
        summary.n_champions = len(champions.allocations)
    finally:
        population.close()
    return summary
