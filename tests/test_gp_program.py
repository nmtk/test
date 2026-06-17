import numpy as np
import pandas as pd

from engine.gp.features import build_features
from engine.gp.operators import OPERATORS
from engine.gp.program import (
    Node,
    crossover,
    depth,
    deserialize,
    evaluate,
    mutate,
    random_tree,
    serialize,
    size,
    to_weight,
)
from engine.gp.strategy import GPStrategy


def _bars(n=200):
    close = pd.Series(100.0 + np.cumsum(np.random.default_rng(0).normal(0, 1, n)))
    return pd.DataFrame(
        {
            "ts": range(n),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1.0,
        }
    )


def test_delay_is_causal():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert OPERATORS["delay1"].fn(s).tolist()[1:] == [1, 2, 3, 4]


def test_safe_div_has_no_inf():
    out = OPERATORS["div"].fn(pd.Series([1.0, 2.0]), pd.Series([0.0, 2.0]))
    assert np.isfinite(out).all()


def test_evaluate_and_weight_bounded():
    feats = build_features(_bars())
    node = Node("op", "add", children=[Node("feat", name="ret"), Node("feat", name="logret")])
    signal = evaluate(node, feats)
    assert len(signal) == 200
    w = to_weight(signal)
    assert ((w >= -1.0) & (w <= 1.0)).all()
    assert not w.isna().any()


def test_random_tree_depth_bounded():
    rng = np.random.default_rng(2)
    for _ in range(30):
        t = random_tree(rng, max_depth=3)
        assert size(t) >= 1
        assert depth(t) <= 4  # max_depth + terminal level


def test_serialize_roundtrip():
    rng = np.random.default_rng(1)
    t = random_tree(rng, max_depth=4)
    assert serialize(deserialize(serialize(t))) == serialize(t)


def test_crossover_and_mutate_valid():
    rng = np.random.default_rng(3)
    a = random_tree(rng, max_depth=4)
    b = random_tree(rng, max_depth=4)
    child = crossover(a, b, rng)
    mutant = mutate(a, rng, max_depth=4)
    feats = build_features(_bars())
    assert size(child) >= 1 and size(mutant) >= 1
    assert len(evaluate(child, feats)) == 200  # still evaluable


def test_gp_strategy_reconstructs_from_serialized():
    program = serialize(Node("op", "ts_zscore20", children=[Node("feat", name="close")]))
    bars = _bars()
    w1 = GPStrategy(program=program).weights(bars)
    w2 = GPStrategy(**{"program": program}).weights(bars)  # reconstruct from stored params
    assert ((w1 >= -1.0) & (w1 <= 1.0)).all()
    assert w1.equals(w2)
