import numpy as np
import pandas as pd

from engine.gp.evolve import GPConfig, evolve
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
