"""macro_flow archetype: the gate gates, direction flips, and absence of context = flat."""

import numpy as np
import pandas as pd
import pytest

from engine.strategies import get_strategy
from engine.strategies.macro_flow import GATE_COLUMNS, MacroFlowTrend

HOUR_MS = 3_600_000


def _bars(n=400, trend=0.002):
    close = 100.0 * np.cumprod(np.full(n, 1.0 + trend))
    return pd.DataFrame(
        {
            "ts": 1_600_000_000_000 + np.arange(n) * HOUR_MS,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
        }
    )


def test_registered():
    assert get_strategy("macro_flow") is MacroFlowTrend


def test_gate_switches_participation_on_and_off():
    n = 400
    bars = _bars(n)  # steady uptrend -> trend signal is +1 once lookback filled
    gate = np.full(n, -1.0)
    gate[n // 2 :] = 1.0  # risk-off first half, risk-on second half
    bars["ctx_risk_on"] = gate

    w = MacroFlowTrend(lookback=20, gate="risk_on", direction=1, threshold=0.2).weights(bars)
    assert (w.iloc[30 : n // 2] == 0.0).all()  # gated out
    assert (w.iloc[n // 2 :] == 1.0).all()  # riding the trend


def test_direction_flips_the_hypothesis():
    n = 400
    bars = _bars(n)
    bars["ctx_dollar_mom"] = -0.8  # dollar falling the whole time
    long_on_weak_dollar = MacroFlowTrend(
        lookback=20, gate="dollar", direction=-1, threshold=0.2
    ).weights(bars)
    long_on_strong_dollar = MacroFlowTrend(
        lookback=20, gate="dollar", direction=1, threshold=0.2
    ).weights(bars)
    assert (long_on_weak_dollar.iloc[30:] == 1.0).all()
    assert (long_on_strong_dollar == 0.0).all()


def test_trend_side_is_preserved_while_gated_in():
    n = 400
    bars = _bars(n, trend=-0.002)  # downtrend -> trend signal -1
    bars["ctx_btc_mom"] = 0.9
    w = MacroFlowTrend(lookback=20, gate="btc", direction=1, threshold=0.0).weights(bars)
    assert (w.iloc[30:] == -1.0).all()


def test_missing_context_column_means_flat():
    bars = _bars(200)  # no ctx_* columns at all
    w = MacroFlowTrend(lookback=20, gate="risk_on").weights(bars)
    assert (w == 0.0).all()


def test_weights_bounded_and_aligned():
    bars = _bars(300)
    bars["ctx_spx_mom"] = np.sin(np.arange(300) / 10.0)
    w = MacroFlowTrend(lookback=15, gate="spx", threshold=0.1).weights(bars)
    assert len(w) == len(bars)
    assert w.abs().max() <= 1.0


def test_unknown_gate_rejected():
    with pytest.raises(ValueError):
        MacroFlowTrend(gate="astrology")


def test_every_declared_gate_has_a_column_mapping():
    space = MacroFlowTrend.param_space
    assert set(space["gate"]) == set(GATE_COLUMNS)


def test_optuna_search_over_categorical_space_runs():
    from engine.search.optuna_search import search

    bars = _bars(350)
    bars["ctx_risk_on"] = 1.0
    bars["ctx_spx_mom"] = 0.5
    for col in GATE_COLUMNS.values():
        if col not in bars.columns:
            bars[col] = 0.0
    outcome = search(bars, MacroFlowTrend, n_trials=5, ppy=8760.0, seed=1)
    assert outcome.best_params["gate"] in GATE_COLUMNS
    assert len(outcome.configs) >= 1
