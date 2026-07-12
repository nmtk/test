"""The alpha pool: one deployable strategy made of many sub-threshold signals.

Six days of production research produced GP programs with DSR ~1.0, SPA p ~0.0 and CPCV
medians of +2..+3 that *still* failed the gate — every one of them on PBO alone. PBO
(CSCV) asks "when you pick the in-sample best of the family, does it stay good out of
sample?"; among a hall of fame of near-equally-good siblings the in-sample winner is
close to a coin flip, so argmax-selection fails PBO *even when every member is genuinely
profitable*. The honest fix is not to loosen the gate but to stop deploying argmax:
combine the decorrelated members into one fixed-weight pool. Diversification raises the
pool's Sharpe above its members' (variance cancels where signals disagree), and because
the pool is a stable object rather than a per-sample selection, its family passes or
fails PBO on real out-of-sample behavior. This is standard multi-alpha practice
(signal blending), made to fit this engine's strategy interface.

Members are frozen at research time — strategy name + params + weight are all stored in
``params`` — so the executor reconstructs the exact validated object with
``get_strategy("ensemble")(**params)`` like any other champion, with no re-estimation
in production.
"""

from __future__ import annotations

import pandas as pd

from engine.strategies.base import Strategy, get_strategy, register


@register
class EnsembleStrategy(Strategy):
    """Fixed weighted sum of member strategies' target weights, clipped to [-1, 1].

    ``members`` is a list of ``{"strategy": name, "params": dict, "weight": float}``.
    ``k`` and ``weighting`` describe how the pool was built (breadth and weighting rule)
    — they are reporting metadata; the ``weight`` values are already resolved.
    """

    name = "ensemble"
    param_space = {}  # built by engine.search.ensemble, not searched by optuna

    def __init__(
        self,
        members: list[dict],
        k: int | None = None,
        weighting: str = "equal",
    ) -> None:
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members = members
        self.k = k if k is not None else len(members)
        self.weighting = weighting

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        total = pd.Series(0.0, index=bars.index)
        for member in self.members:
            strategy = get_strategy(member["strategy"])(**member["params"])
            w = strategy.weights(bars).astype(float).fillna(0.0)
            total = total + float(member["weight"]) * w
        return total.clip(-1.0, 1.0)

    @property
    def params(self) -> dict:
        return {"members": self.members, "k": self.k, "weighting": self.weighting}

    def describe(self) -> str:
        names = ", ".join(m["strategy"] for m in self.members)
        return f"ensemble(k={self.k}, weighting={self.weighting}, members=[{names}])"
