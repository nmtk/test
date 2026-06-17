from engine.orchestration.population import Population
from engine.search.research import run_research
from engine.validation.gate import GateThresholds

# Ultra-lenient gate so the discovery->incubation plumbing is exercised deterministically
# (the strict gate's noise-rejection is covered in test_validation_gate).
LENIENT = GateThresholds(
    dsr_min=-1.0, pbo_max=1.0, spa_max=1.0, min_oos_sharpe=-1e9, min_pct_positive=0.0
)


def test_run_research_incubates_passing_strategies(paper_settings):
    summary = run_research(
        paper_settings,
        strategies=("tsmom",),
        symbols=["BTC/JPY"],
        timeframes=["1h"],
        trials=6,
        thresholds=LENIENT,
        min_bars=150,
    )
    assert summary.n_evaluated == 1
    assert summary.n_passed == 1  # lenient gate passes

    pop = Population(paper_settings.state_dir / "population.db")
    rec = pop.get("tsmom:BTC/JPY:1h")
    assert rec is not None
    # passed strategies enter incubation (NOT promoted straight to champion)
    assert rec["status"] == "incubating"
    assert rec["incubation_start_ts"] is not None
    # backtest reference stats are recorded for the forward-test bands
    assert "sharpe_per_bar" in rec["metrics"]
    assert "ppy" in rec["metrics"]
    pop.close()


def test_run_research_records_failures_as_candidates(paper_settings):
    # Impossible thresholds -> nothing passes -> recorded as candidate, never incubating.
    strict = GateThresholds(dsr_min=2.0)  # DSR is a probability in [0,1]; never reachable
    summary = run_research(
        paper_settings,
        strategies=("tsmom",),
        symbols=["BTC/JPY"],
        timeframes=["1h"],
        trials=6,
        thresholds=strict,
        min_bars=150,
    )
    assert summary.n_passed == 0
    pop = Population(paper_settings.state_dir / "population.db")
    assert pop.get("tsmom:BTC/JPY:1h")["status"] == "candidate"
    pop.close()
