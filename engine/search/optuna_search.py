"""optuna parameter search and challenger discovery.

``search`` runs TPE optimization over a strategy's parameter space using the robust score.
``discover_challenger`` then takes the explored parameter family, runs it through the M2
validation gate (counting the trials in the ledger so DSR is honestly deflated), and
returns the resulting report — a challenger iff it passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import optuna

from engine.search.objective import DEFAULT_TURNOVER_PENALTY, robust_score
from engine.search.space import suggest_params
from engine.validation.gate import (
    GateThresholds,
    ValidationReport,
    evaluate_matrix,
    returns_for_configs,
)
from engine.validation.ledger import TrialLedger

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(slots=True)
class SearchOutcome:
    best_params: dict
    best_score: float
    configs: list[dict] = field(default_factory=list)  # distinct explored parameter sets


def search(
    bars,
    strategy_cls,
    *,
    n_trials: int = 40,
    ppy: float = 365.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    seed: int = 0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
) -> SearchOutcome:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, strategy_cls.param_space)
        return robust_score(
            bars,
            strategy_cls,
            params,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            ppy=ppy,
            turnover_penalty=turnover_penalty,
        )

    study.optimize(objective, n_trials=n_trials)

    seen: set = set()
    configs: list[dict] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        key = tuple(sorted(t.params.items()))
        if key not in seen:
            seen.add(key)
            configs.append(dict(t.params))
    return SearchOutcome(
        best_params=study.best_params, best_score=study.best_value, configs=configs
    )


def discover_challenger(
    bars,
    strategy_cls,
    *,
    n_trials: int = 40,
    ppy: float = 365.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    seed: int = 0,
    thresholds: GateThresholds | None = None,
    ledger: TrialLedger | None = None,
    family: str | None = None,
    inherited_trials: int = 0,
    **gate_kwargs,
) -> ValidationReport:
    """Search, then validate the explored family through the gate; returns its report.

    ``inherited_trials`` adds another family's cumulative count to the DSR deflation, for
    a family that re-searches a space an earlier one already mined.
    """
    outcome = search(
        bars,
        strategy_cls,
        n_trials=n_trials,
        ppy=ppy,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        seed=seed,
    )
    configs = outcome.configs
    matrix = returns_for_configs(
        bars, strategy_cls, configs, fee_bps=fee_bps, slippage_bps=slippage_bps, ppy=ppy
    )

    fam = family or getattr(strategy_cls, "name", "global")
    trials = len(configs)
    if ledger is not None:
        trials = ledger.add(fam, len(configs))

    # Same selection rule the gate uses internally, so best_params matches the scored column.
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_bar_sharpe = np.where(std > 0, mean / std, 0.0)
    best_params = configs[int(np.argmax(per_bar_sharpe))]

    return evaluate_matrix(
        matrix,
        name=fam,
        best_params=best_params,
        ppy=ppy,
        thresholds=thresholds,
        n_trials=trials + int(inherited_trials),
        **gate_kwargs,
    )
