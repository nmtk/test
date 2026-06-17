"""GP program trees: random generation, evaluation, variation, and (de)serialization.

A program is a tree of operator nodes over feature/constant terminals that evaluates to a
per-bar *signal* series. The signal is turned into a target weight in [-1, 1] by a causal
volatility squash, so program scale doesn't matter and there is no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.gp.features import FEATURE_NAMES
from engine.gp.operators import BINARY_OPS, OPERATORS, UNARY_OPS

_ALL_OPS = (*BINARY_OPS, *UNARY_OPS)


@dataclass(slots=True)
class Node:
    kind: str  # "op" | "feat" | "const"
    name: str = ""
    value: float = 0.0
    children: list = field(default_factory=list)


# -- construction -----------------------------------------------------------------------


def _random_terminal(rng: np.random.Generator) -> Node:
    if rng.random() < 0.7:
        return Node("feat", name=str(rng.choice(FEATURE_NAMES)))
    return Node("const", value=float(rng.uniform(-2.0, 2.0)))


def random_tree(
    rng: np.random.Generator, max_depth: int = 4, depth: int = 0, terminal_prob: float = 0.3
) -> Node:
    """Grow a random program tree (depth-limited)."""
    if depth >= max_depth or (depth > 0 and rng.random() < terminal_prob):
        return _random_terminal(rng)
    op = str(rng.choice(_ALL_OPS))
    arity = OPERATORS[op].arity
    children = [random_tree(rng, max_depth, depth + 1, terminal_prob) for _ in range(arity)]
    return Node("op", name=op, children=children)


# -- evaluation -------------------------------------------------------------------------


def evaluate(node: Node, features: dict[str, pd.Series]) -> pd.Series:
    """Evaluate a program to a signal series over the feature index."""
    if node.kind == "feat":
        return features[node.name]
    if node.kind == "const":
        index = features[next(iter(features))].index
        return pd.Series(node.value, index=index, dtype=float)
    args = [evaluate(c, features) for c in node.children]
    out = OPERATORS[node.name].fn(*args)
    return pd.Series(out).replace([np.inf, -np.inf], np.nan)


def to_weight(signal: pd.Series, vol_window: int = 100) -> pd.Series:
    """Causally squash an arbitrary-scale signal into a target weight in [-1, 1]."""
    std = signal.rolling(vol_window, min_periods=10).std(ddof=0)
    normed = signal / std.replace(0, np.nan)
    return np.tanh(normed.fillna(0.0)).clip(-1.0, 1.0)


# -- structure / variation --------------------------------------------------------------


def size(node: Node) -> int:
    return 1 + sum(size(c) for c in node.children)


def depth(node: Node) -> int:
    return 1 + (max((depth(c) for c in node.children), default=0))


def clone(node: Node) -> Node:
    return Node(node.kind, node.name, node.value, [clone(c) for c in node.children])


def _collect(node: Node) -> list[Node]:
    nodes = [node]
    for c in node.children:
        nodes.extend(_collect(c))
    return nodes


def _overwrite(target: Node, donor: Node) -> None:
    target.kind, target.name, target.value, target.children = (
        donor.kind,
        donor.name,
        donor.value,
        donor.children,
    )


def crossover(a: Node, b: Node, rng: np.random.Generator) -> Node:
    """Replace a random subtree of a copy of ``a`` with a random subtree of ``b``."""
    child = clone(a)
    target = rng.choice(_collect(child))
    donor = clone(rng.choice(_collect(b)))
    _overwrite(target, donor)
    return child


def mutate(a: Node, rng: np.random.Generator, max_depth: int = 4) -> Node:
    """Replace a random subtree of a copy of ``a`` with a fresh random subtree."""
    child = clone(a)
    target = rng.choice(_collect(child))
    _overwrite(target, random_tree(rng, max_depth=max_depth))
    return child


# -- serialization ----------------------------------------------------------------------


def serialize(node: Node) -> list:
    if node.kind == "feat":
        return ["feat", node.name]
    if node.kind == "const":
        return ["const", node.value]
    return ["op", node.name, [serialize(c) for c in node.children]]


def deserialize(data: list) -> Node:
    kind = data[0]
    if kind == "feat":
        return Node("feat", name=data[1])
    if kind == "const":
        return Node("const", value=float(data[1]))
    return Node("op", name=data[1], children=[deserialize(c) for c in data[2]])


def to_str(node: Node) -> str:
    if node.kind == "feat":
        return node.name
    if node.kind == "const":
        return f"{node.value:.3g}"
    return f"{node.name}(" + ", ".join(to_str(c) for c in node.children) + ")"
