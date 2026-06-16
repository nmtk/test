import numpy as np
import pytest

from engine.validation.splits import combinatorial_purged_splits, n_backtest_paths


def test_path_count():
    assert n_backtest_paths(6, 2) == 5  # C(6,2)*2/6
    assert n_backtest_paths(5, 2) == 4  # C(5,2)*2/5


def test_split_count_and_no_overlap():
    splits = combinatorial_purged_splits(100, n_groups=5, n_test_groups=2, embargo_frac=0.0)
    assert len(splits) == 10  # C(5, 2)
    for s in splits:
        assert set(s.train_idx).isdisjoint(set(s.test_idx))
        assert set(s.train_idx) | set(s.test_idx) <= set(range(100))
        assert len(s.test_idx) > 0


def test_embargo_and_purge_shrink_train():
    no_embargo = combinatorial_purged_splits(200, 5, 2, embargo_frac=0.0, purge_gap=0)
    with_embargo = combinatorial_purged_splits(200, 5, 2, embargo_frac=0.1, purge_gap=2)
    # same test sets, but more training data removed when purge+embargo are on
    assert len(no_embargo[0].train_idx) > len(with_embargo[0].train_idx)


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        combinatorial_purged_splits(100, n_groups=3, n_test_groups=3)
    with pytest.raises(ValueError):
        combinatorial_purged_splits(2, n_groups=5, n_test_groups=2)


def test_purge_removes_boundary_bars():
    # one test group [40,60); with purge_gap=3 the bars 37,38,39 and 60,61,62 leave train
    splits = combinatorial_purged_splits(
        100, n_groups=5, n_test_groups=1, embargo_frac=0.0, purge_gap=3
    )
    first = next(s for s in splits if s.test_groups == (2,))
    train = set(first.train_idx)
    assert {37, 38, 39}.isdisjoint(train)
    assert {60, 61, 62}.isdisjoint(train)
    assert 36 in train and 63 in train
    np.testing.assert_array_equal(first.test_idx, np.arange(40, 60))
