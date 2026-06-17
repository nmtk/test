from engine.orchestration.champions import load_champions
from engine.orchestration.population import Population, StrategyRecord
from engine.search.research import build_champions_from_population, run_research
from engine.validation.gate import GateThresholds

# Ultra-lenient gate so the promotion plumbing is exercised deterministically (the strict
# gate's noise-rejection is already covered in test_validation_gate).
LENIENT = GateThresholds(
    dsr_min=-1.0, pbo_max=1.0, spa_max=1.0, min_oos_sharpe=-1e9, min_pct_positive=0.0
)


def _rec(family, dsr, status="challenger"):
    return StrategyRecord(
        family=family,
        strategy="tsmom",
        symbol=family.split(":")[1],
        source="bitbank",
        timeframe="1h",
        params={"lookback": 50},
        status=status,
        metrics={"dsr": dsr, "cpcv_median": dsr},
    )


def test_build_champions_selects_top_and_demotes(tmp_path):
    pop = Population(tmp_path / "pop.db")
    pop.upsert(_rec("tsmom:BTC/JPY:1h", 0.99))
    pop.upsert(_rec("tsmom:ETH/JPY:1h", 0.80))
    pop.upsert(_rec("tsmom:XRP/JPY:1h", 0.60))

    champs = build_champions_from_population(pop, max_champions=2, gross=1.0)
    assert len(champs.allocations) == 2
    symbols = {a.symbol for a in champs.allocations}
    assert symbols == {"BTC/JPY", "ETH/JPY"}  # top-2 by dsr
    assert all(abs(a.weight - 0.5) < 1e-9 for a in champs.allocations)  # equal weight
    # the unselected one is demoted back to challenger
    assert len(pop.list("champion")) == 2
    assert len(pop.list("challenger")) == 1
    pop.close()


def test_run_research_end_to_end(paper_settings):
    summary = run_research(
        paper_settings,
        strategies=("tsmom",),
        symbols=["BTC/JPY"],
        timeframes=["1h"],
        trials=6,
        thresholds=LENIENT,
        min_bars=150,
        max_champions=4,
    )
    assert summary.n_evaluated == 1
    assert summary.n_passed == 1  # lenient gate passes
    assert summary.n_champions == 1

    champs = load_champions(paper_settings.state_dir / "champions.json")
    assert len(champs.allocations) == 1
    assert champs.allocations[0].strategy == "tsmom"
    assert champs.allocations[0].symbol == "BTC/JPY"
    assert (paper_settings.state_dir / "population.db").exists()
