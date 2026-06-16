"""Strategy interface, parameter-space description, and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class IntParam:
    low: int
    high: int
    step: int = 1


@dataclass(frozen=True, slots=True)
class FloatParam:
    low: float
    high: float
    log: bool = False


# A parameter space maps a parameter name to its search range. Consumed by the optuna
# search in M4; defined here so each archetype declares its own tunable surface.
ParamSpace = dict[str, IntParam | FloatParam | list]


class Strategy(ABC):
    """Maps bars -> target weight in [-1, 1] per bar (causal)."""

    name: str = "base"
    param_space: ParamSpace = {}

    @abstractmethod
    def weights(self, bars: pd.DataFrame) -> pd.Series:
        """Return a target-weight series aligned to ``bars`` (values in [-1, 1])."""

    @property
    def params(self) -> dict:
        """The concrete parameters of this instance (for logging/serialization)."""
        return {k: getattr(self, k) for k in self.param_space}

    def describe(self) -> str:
        kv = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({kv})"


_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"duplicate strategy name: {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> type[Strategy]:
    return _REGISTRY[name]


def all_strategies() -> dict[str, type[Strategy]]:
    return dict(_REGISTRY)


class CrossSectionalStrategy(ABC):
    """Maps a wide close panel (time x symbols) -> a wide target-weight panel.

    Cross-sectional archetypes rank instruments against each other each bar (e.g. long the
    strongest, short the weakest), so they need the whole basket at once rather than one
    series. Each output row holds signed target weights with ``sum(|w|) <= 1`` (gross
    exposure capped at 1); dollar-neutral books sum to ~0.
    """

    name: str = "xs_base"
    param_space: ParamSpace = {}

    @abstractmethod
    def weights_panel(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Return a target-weight panel aligned to ``closes`` (index=time, cols=symbols)."""

    @property
    def params(self) -> dict:
        return {k: getattr(self, k) for k in self.param_space}

    def describe(self) -> str:
        kv = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({kv})"


_XS_REGISTRY: dict[str, type[CrossSectionalStrategy]] = {}


def register_xs(cls: type[CrossSectionalStrategy]) -> type[CrossSectionalStrategy]:
    if cls.name in _XS_REGISTRY and _XS_REGISTRY[cls.name] is not cls:
        raise ValueError(f"duplicate cross-sectional strategy name: {cls.name!r}")
    _XS_REGISTRY[cls.name] = cls
    return cls


def get_xs_strategy(name: str) -> type[CrossSectionalStrategy]:
    return _XS_REGISTRY[name]


def all_xs_strategies() -> dict[str, type[CrossSectionalStrategy]]:
    return dict(_XS_REGISTRY)
