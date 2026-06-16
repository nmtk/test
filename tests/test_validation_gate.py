"""The overfitting self-test: the gate MUST reject pure noise and accept a real edge.

This is the proof the validation actually validates — if noise ever passed, the whole
premise (a rigorous gate guarding against data mining) would be hollow.
"""

import numpy as np
import pandas as pd

from engine.strategies import TimeSeriesMomentum
from engine.validation.gate import (
    GateThresholds,
    evaluate_candidate_family,
    evaluate_matrix,
)
from engine.validation.ledger import TrialLedger


def _noise_matrix(seed, t=500, c=20):
    return np.random.default_rng(seed).normal(0.0, 0.01, size=(t, c))


def _signal_matrix(seed, t=500, c=20):
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, 0.01, size=(t, c))
    m[:, 0] = rng.normal(0.01, 0.01, size=t)  # one genuine, persistent edge
    return m


def test_noise_is_rejected():
    report = evaluate_matrix(_noise_matrix(0), ppy=365.0, n_trials=20, thresholds=GateThresholds())
    assert report.passed is False
    assert report.dsr < 0.95  # not significant after deflation
    # the failure shows up as overfitting and/or insignificance
    assert (report.checks["pbo"] is False) or (report.checks["dsr"] is False)


def test_genuine_edge_is_accepted():
    report = evaluate_matrix(_signal_matrix(0), ppy=365.0, n_trials=20, thresholds=GateThresholds())
    assert report.passed is True
    assert report.dsr >= 0.95
    assert report.pbo <= 0.2
    assert report.spa_p <= 0.05
    assert report.cpcv_median > 0.0


def test_ledger_roundtrip(tmp_path):
    path = tmp_path / "trial_ledger.json"
    led = TrialLedger.load(path)
    assert led.total() == 0
    led.add("tsmom", 5)
    led.add("tsmom", 3)
    led.save()
    again = TrialLedger.load(path)
    assert again.get("tsmom") == 8
    assert again.total() == 8


def test_candidate_family_accumulates_trials(tmp_path):
    n = 400
    close = pd.Series(100.0 * (1.0008 ** np.arange(n)))
    bars = pd.DataFrame(
        {"ts": range(n), "open": close, "high": close, "low": close, "close": close, "volume": 1.0}
    )
    grid = {"lookback": [10, 20, 50]}
    led = TrialLedger.load(tmp_path / "trial_ledger.json")

    r1 = evaluate_candidate_family(
        bars, TimeSeriesMomentum, grid, ppy=365.0, ledger=led, family="t"
    )
    assert r1.n_configs == 3
    assert r1.n_trials == 3  # cumulative after first run

    r2 = evaluate_candidate_family(
        bars, TimeSeriesMomentum, grid, ppy=365.0, ledger=led, family="t"
    )
    assert r2.n_trials == 6  # honestly deflated by all trials ever run
    assert led.get("t") == 6
