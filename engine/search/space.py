"""Map a strategy's declared parameter space onto optuna trial suggestions."""

from __future__ import annotations

from engine.strategies.base import FloatParam, IntParam, ParamSpace


def suggest_params(trial, param_space: ParamSpace) -> dict:
    """Draw one parameter set from ``param_space`` using an optuna ``trial``."""
    params: dict = {}
    for name, spec in param_space.items():
        if isinstance(spec, IntParam):
            params[name] = trial.suggest_int(name, spec.low, spec.high, step=spec.step)
        elif isinstance(spec, FloatParam):
            params[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
        elif isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
        else:
            raise TypeError(f"unsupported param spec for {name!r}: {type(spec)!r}")
    return params
