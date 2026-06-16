import numpy as np
import pytest

from engine.validation.pbo import probability_of_backtest_overfitting


def test_pure_noise_has_high_pbo():
    rng = np.random.default_rng(0)
    matrix = rng.normal(0.0, 0.01, size=(600, 20))  # no config has a real edge
    pbo = probability_of_backtest_overfitting(matrix, n_partitions=10).pbo
    # selecting on noise overfits: the IS winner regresses below the OOS median >= half
    # the time, so PBO is high.
    assert pbo >= 0.4


def test_genuine_winner_has_low_pbo():
    rng = np.random.default_rng(1)
    matrix = rng.normal(0.0, 0.01, size=(600, 20))
    matrix[:, 0] = rng.normal(0.02, 0.01, size=600)  # one real, persistent edge
    pbo = probability_of_backtest_overfitting(matrix, n_partitions=10).pbo
    assert pbo <= 0.1


def test_requires_multiple_configs():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((100, 5)), n_partitions=7)
