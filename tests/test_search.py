import numpy as np
import pandas as pd
from optuna.trial import FixedTrial

from engine.search.objective import robust_score
from engine.search.optuna_search import discover_challenger, search
from engine.search.space import suggest_params
from engine.strategies import TimeSeriesMomentum
from engine.validation.gate import GateThresholds, ValidationReport
from engine.validation.ledger import TrialLedger


def _bars(n=250):
    close = pd.Series(100.0 * (1.005 ** np.arange(n)) + np.sin(np.arange(n)))
    return pd.DataFrame(
        {"ts": range(n), "open": close, "high": close, "low": close, "close": close, "volume": 1.0}
    )


def test_suggest_params_maps_space():
    params = suggest_params(FixedTrial({"lookback": 42}), TimeSeriesMomentum.param_space)
    assert params == {"lookback": 42}


def test_robust_score_is_finite():
    score = robust_score(_bars(), TimeSeriesMomentum, {"lookback": 30}, ppy=365.0)
    assert np.isfinite(score)


def test_search_is_deterministic():
    a = search(_bars(), TimeSeriesMomentum, n_trials=8, seed=1)
    b = search(_bars(), TimeSeriesMomentum, n_trials=8, seed=1)
    assert a.best_params == b.best_params
    assert len(a.configs) > 0
    assert 5 <= a.best_params["lookback"] <= 200


def test_discover_challenger_counts_trials_in_ledger(tmp_path):
    led = TrialLedger.load(tmp_path / "trial_ledger.json")
    report = discover_challenger(
        _bars(), TimeSeriesMomentum, n_trials=8, ppy=365.0, seed=0, ledger=led, family="f"
    )
    assert isinstance(report, ValidationReport)
    assert "lookback" in report.best_params
    assert report.n_trials == led.get("f")  # deflation uses the cumulative ledger count
    assert report.n_trials > 0


def test_discover_challenger_runs_gate():
    report = discover_challenger(
        _bars(), TimeSeriesMomentum, n_trials=8, ppy=365.0, seed=0, thresholds=GateThresholds()
    )
    # full gate metrics are populated regardless of pass/fail
    assert 0.0 <= report.pbo <= 1.0
    assert 0.0 <= report.dsr <= 1.0
    assert set(report.checks) == {"dsr", "pbo", "spa", "cpcv_median", "cpcv_pct_positive"}


def test_every_registered_archetype_is_actually_researched():
    """A registered strategy the nightly loop never runs is dead code that looks alive.

    trade_balance shipped registered, configured, and with a research-only deep-history
    series wired for it — and was still never researched, because it was missing from
    this one tuple. Nothing failed; it simply never appeared in the population.
    """
    import engine.strategies as strategies_pkg
    from engine.search.research import DEFAULT_ARCHETYPES
    from engine.strategies.base import Strategy

    # Derived from what the package explicitly exports, not the global registry: test
    # modules register dummy strategies into that registry, which would make this pass or
    # fail depending on collection order.
    shipped = {
        obj.name
        for obj in (getattr(strategies_pkg, n) for n in strategies_pkg.__all__)
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy
    }
    # Generated/composite archetypes are driven by their own discovery paths, not the
    # per-archetype parameter search.
    driven_elsewhere = {"gp", "ensemble"}
    missing = shipped - driven_elsewhere - set(DEFAULT_ARCHETYPES)
    assert not missing, f"shipped but never researched: {sorted(missing)}"
