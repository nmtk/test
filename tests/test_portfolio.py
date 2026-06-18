import numpy as np
import pandas as pd
import pytest

from engine.portfolio.allocate import AllocationConfig, AllocInput, allocate
from engine.portfolio.regime import RegimeConfig, exposure_scale

OPEN = AllocationConfig(max_weight=1.0, gp_family_cap=1.0)  # no caps: test raw proportions


def _inp(family, vol=0.01, mean=0.001, se=0.0003, is_gp=False):
    return AllocInput(family=family, vol=vol, mean=mean, se=se, is_gp=is_gp)


def test_symmetric_inputs_get_equal_weight():
    w = allocate([_inp("a"), _inp("b")], None, OPEN)
    assert abs(w["a"] - w["b"]) < 1e-9
    assert abs(w["a"] + w["b"] - 1.0) < 1e-9


def test_risk_parity_favors_lower_vol():
    w = allocate([_inp("a", vol=0.01), _inp("b", vol=0.02)], None, OPEN)
    assert w["a"] > w["b"]
    assert w["a"] == pytest.approx(2 * w["b"], rel=1e-6)


def test_uncertain_or_losing_is_shrunk_to_zero():
    w = allocate(
        [_inp("good", mean=0.001, se=0.0003), _inp("bad", mean=-0.001, se=0.0003)], None, OPEN
    )
    assert w["bad"] == 0.0
    assert w["good"] > 0.0


def test_correlation_penalty_downweights_clustered():
    corr = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    w = allocate([_inp("a"), _inp("b"), _inp("c")], corr, OPEN)
    assert w["c"] > w["a"]  # the uncorrelated name earns more than each of the correlated pair
    assert abs(w["a"] - w["b"]) < 1e-9


def test_gp_family_cap_enforced():
    cfg = AllocationConfig(max_weight=1.0, gp_family_cap=0.2)
    w = allocate([_inp("gp:1", is_gp=True), _inp("gp:2", is_gp=True), _inp("tsmom:1")], None, cfg)
    assert w["gp:1"] + w["gp:2"] <= 0.2 + 1e-9


def test_per_strategy_cap_enforced():
    cfg = AllocationConfig(max_weight=0.34, gp_family_cap=1.0)
    w = allocate([_inp("solo")], None, cfg)
    assert w["solo"] == pytest.approx(0.34)


def test_regime_scale_calm_is_near_one():
    calm = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, 200)))
    assert exposure_scale(calm, RegimeConfig()) >= 0.8  # steady vol -> little/no damping


def test_regime_scale_turbulent_is_below_one():
    rng = np.random.default_rng(1)
    series = list(100.0 + np.cumsum(rng.normal(0, 0.1, 180)))
    series += list(series[-1] + np.cumsum(rng.normal(0, 5.0, 40)))  # turbulent tail
    assert exposure_scale(pd.Series(series), RegimeConfig()) < 1.0


def test_regime_scale_short_series_is_one():
    assert exposure_scale(pd.Series([100.0, 101.0, 102.0]), RegimeConfig()) == 1.0
