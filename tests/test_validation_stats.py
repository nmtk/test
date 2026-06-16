import math

import numpy as np
import pytest

from engine.validation.stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    return_moments,
)


def test_psr_half_when_equal_to_benchmark():
    assert probabilistic_sharpe_ratio(0.1, 1000, 0.0, 3.0, benchmark_sharpe=0.1) == pytest.approx(
        0.5, abs=1e-9
    )


def test_psr_high_for_strong_sharpe():
    assert probabilistic_sharpe_ratio(0.1, 1000, 0.0, 3.0, benchmark_sharpe=0.0) > 0.99


def test_psr_negative_skew_lowers_confidence():
    base = probabilistic_sharpe_ratio(0.1, 500, 0.0, 3.0)
    skewed = probabilistic_sharpe_ratio(0.1, 500, -1.0, 6.0)  # fat left tail
    assert skewed < base


def test_expected_max_sharpe_increases_with_trials():
    assert expected_max_sharpe(0.04, 2) < expected_max_sharpe(0.04, 100)
    assert expected_max_sharpe(0.0, 100) == 0.0
    assert expected_max_sharpe(0.04, 1) == 0.0


def test_deflated_below_plain_psr_when_many_trials():
    psr = probabilistic_sharpe_ratio(0.12, 800, 0.0, 3.0)
    dsr = deflated_sharpe_ratio(0.12, 800, 0.0, 3.0, sharpe_variance=0.04, n_trials=200)
    assert dsr < psr


def test_min_track_record_length():
    assert math.isinf(min_track_record_length(0.05, 0.0, 3.0, benchmark_sharpe=0.05))
    near = min_track_record_length(0.06, 0.0, 3.0, benchmark_sharpe=0.05)
    far = min_track_record_length(0.20, 0.0, 3.0, benchmark_sharpe=0.05)
    assert far < near  # a bigger edge needs fewer observations to prove


def test_return_moments_known():
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.01])
    m = return_moments(r)
    assert m.n == 5
    assert m.sharpe == pytest.approx(r.mean() / r.std(ddof=1))
