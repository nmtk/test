import numpy as np
import pandas as pd

from engine.gp.evolve import GPConfig, evolve, max_abs_correlation
from engine.gp.features import build_features
from engine.gp.fitness import program_weights
from engine.gp.program import deserialize
from engine.search.gp_search import discover_gp
from engine.validation.gate import GateThresholds, ValidationReport
from engine.validation.ledger import TrialLedger

CFG = GPConfig(pop_size=12, generations=3, hall_of_fame=6)


def _bars(n=300, seed=0):
    steps = np.random.default_rng(seed).normal(0.001, 0.01, n)
    close = pd.Series(100.0 * np.exp(np.cumsum(steps)))
    return pd.DataFrame(
        {
            "ts": range(n),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
        }
    )


def test_evolve_is_deterministic():
    a = evolve(_bars(), ppy=365.0, cfg=CFG, seed=1)
    b = evolve(_bars(), ppy=365.0, cfg=CFG, seed=1)
    assert a.n_evaluated > 0
    assert len(a.hall_of_fame) >= 1
    assert a.hall_of_fame[0][0] == b.hall_of_fame[0][0]  # same best program (serialized)


def test_max_abs_correlation_edges():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    assert max_abs_correlation(a, []) == 0.0
    assert max_abs_correlation(a, [a.copy()]) > 0.999  # identical = same alpha
    assert max_abs_correlation(a, [-a]) > 0.999  # sign flip is still the same bet
    constant = np.zeros(200)
    assert max_abs_correlation(constant, [np.ones(200)]) == 1.0  # constants = duplicates
    assert max_abs_correlation(constant, [a]) == 0.0  # constant vs varying: no info


def test_hall_of_fame_members_are_decorrelated():
    bars = _bars(n=400, seed=3)
    result = evolve(bars, ppy=365.0, cfg=CFG, seed=3)
    features = build_features(bars)
    streams = [
        np.nan_to_num(
            program_weights(deserialize(prog), features).to_numpy(dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for prog, _ in result.hall_of_fame
    ]
    for i, stream in enumerate(streams):
        assert max_abs_correlation(stream, streams[:i]) <= CFG.hof_corr_cap + 1e-9


def test_discover_gp_runs_gate_and_counts_trials(tmp_path):
    led = TrialLedger.load(tmp_path / "ledger.json")
    report = discover_gp(
        _bars(),
        ppy=365.0,
        cfg=CFG,
        seed=0,
        ledger=led,
        family="gp:BTC/JPY:1h",
        thresholds=GateThresholds(),
    )
    assert isinstance(report, ValidationReport)
    assert "program" in report.best_params
    # the ledger counts ALL evolved programs (>= hall-of-fame size), deflating DSR honestly
    assert report.n_trials == led.get("gp:BTC/JPY:1h")
    assert report.n_trials >= len(report.best_params)  # sanity: trials counted
    assert 0.0 <= report.dsr <= 1.0
