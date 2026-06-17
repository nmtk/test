import numpy as np

from engine.monitor.decay import cusum_down, page_hinkley, sharpe_lower_band


def test_page_hinkley_ignores_stable_noise():
    stable = np.random.default_rng(0).normal(0.0, 1.0, 300)
    assert not page_hinkley(stable, delta=0.5, threshold=8.0)


def test_page_hinkley_detects_downshift():
    rng = np.random.default_rng(1)
    shifted = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(-3.0, 1.0, 100)])
    assert page_hinkley(shifted, delta=0.5, threshold=8.0)


def test_cusum_down():
    assert not cusum_down(np.full(100, 0.1), target=0.0, drift=0.5, threshold=5.0)
    assert cusum_down(np.full(100, -2.0), target=0.0, drift=0.5, threshold=5.0)


def test_sharpe_lower_band():
    band = sharpe_lower_band(0.1, 500, 0.0, 3.0, z=1.645)
    assert band < 0.1  # lower bound sits below the point estimate
    assert sharpe_lower_band(0.1, 1, 0.0, 3.0) == float("-inf")  # undefined for tiny n
