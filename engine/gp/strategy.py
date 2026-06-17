"""Wrap an evolved GP program as a standard ``Strategy`` so it flows through the whole engine.

Registered under the name ``gp``; its single parameter is the serialized program, so the
population/executor reconstruct it with ``get_strategy("gp")(program=...)`` exactly like any
other strategy.
"""

from __future__ import annotations

import pandas as pd

from engine.gp.features import build_features
from engine.gp.program import deserialize, evaluate, to_str, to_weight
from engine.strategies.base import Strategy, register


@register
class GPStrategy(Strategy):
    name = "gp"
    param_space = {}  # GP has its own (evolutionary) search, not an optuna grid

    def __init__(self, program: list) -> None:
        self.program = program  # serialized nested-list form (see engine.gp.program)
        self._node = deserialize(program)

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        signal = evaluate(self._node, build_features(bars))
        weights = to_weight(signal)
        weights.index = bars.index
        return weights.fillna(0.0).astype(float)

    @property
    def params(self) -> dict:
        return {"program": self.program}

    def describe(self) -> str:
        return f"gp[{to_str(self._node)}]"
