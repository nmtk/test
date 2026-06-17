"""Causal operators for GP alpha programs.

Every operator is a function of one or two pandas Series returning a Series, and is *causal*
(uses only the current and past bars — never the future), so an evolved program can be
evaluated bar-by-bar in live trading exactly as in research. Windowed variants are baked at
fixed lookbacks to keep every node series-typed (no integer arguments in the tree).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class Operator:
    name: str
    arity: int
    fn: Callable


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series(np.where(np.abs(b) > _EPS, a / b.replace(0, np.nan), 1.0), index=a.index)


def _binary() -> list[Operator]:
    return [
        Operator("add", 2, lambda a, b: a + b),
        Operator("sub", 2, lambda a, b: a - b),
        Operator("mul", 2, lambda a, b: a * b),
        Operator("div", 2, _safe_div),
    ]


def _unary() -> list[Operator]:
    ops = [
        Operator("neg", 1, lambda a: -a),
        Operator("abs", 1, lambda a: a.abs()),
        Operator("sign", 1, lambda a: np.sign(a)),
        Operator("tanh", 1, lambda a: np.tanh(a)),
        Operator("delay1", 1, lambda a: a.shift(1)),
        Operator("delay5", 1, lambda a: a.shift(5)),
        Operator("delta1", 1, lambda a: a.diff(1)),
        Operator("delta5", 1, lambda a: a.diff(5)),
    ]
    for w in (5, 10, 20, 50):
        ops.append(Operator(f"ts_mean{w}", 1, lambda a, w=w: a.rolling(w, min_periods=w).mean()))
        ops.append(
            Operator(f"ts_std{w}", 1, lambda a, w=w: a.rolling(w, min_periods=w).std(ddof=0))
        )
        ops.append(Operator(f"ts_zscore{w}", 1, _ts_zscore(w)))
        ops.append(
            Operator(f"ts_rank{w}", 1, lambda a, w=w: a.rolling(w, min_periods=w).rank(pct=True))
        )
    return ops


def _ts_zscore(window: int) -> Callable:
    def fn(a: pd.Series) -> pd.Series:
        mean = a.rolling(window, min_periods=window).mean()
        std = a.rolling(window, min_periods=window).std(ddof=0)
        return (a - mean) / std.replace(0, np.nan)

    return fn


OPERATORS: dict[str, Operator] = {op.name: op for op in (*_binary(), *_unary())}
BINARY_OPS: tuple[str, ...] = tuple(op.name for op in _binary())
UNARY_OPS: tuple[str, ...] = tuple(op.name for op in _unary())
