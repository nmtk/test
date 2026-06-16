import numpy as np

from engine.validation.spa import spa_test, stationary_bootstrap_indices


def test_bootstrap_indices_shape_and_range():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(50, avg_block=10, n_boot=100, rng=rng)
    assert idx.shape == (100, 50)
    assert idx.min() >= 0 and idx.max() < 50


def test_superior_model_rejects_null():
    rng = np.random.default_rng(0)
    perf = rng.normal(0.0, 0.01, size=(500, 10))
    perf[:, 0] = rng.normal(0.01, 0.01, size=500)  # genuinely beats the benchmark
    res = spa_test(perf, n_boot=300, seed=0)
    assert res.p_value < 0.05
    assert res.statistic > 0.0


def test_pure_noise_does_not_reject():
    rng = np.random.default_rng(2)
    perf = rng.normal(0.0, 0.01, size=(500, 10))  # nothing beats the benchmark
    res = spa_test(perf, n_boot=300, seed=0)
    assert res.p_value > 0.05
