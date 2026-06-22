"""Bridge GP evolution to the validation gate (mirror of optuna_search.discover_challenger).

Evolves a family of alpha programs, backtests the hall-of-fame into a returns matrix, and
runs it through the same gate. Every distinct program evaluated during evolution is counted
in the trial ledger, so the Deflated Sharpe Ratio is deflated by the (large) number of GP
trials — GP's extra overfitting risk is paid for honestly.

Pass ``hof_path`` to persist the hall-of-fame across nightly runs. The next call will
warm-start the population from the survivors, accumulating GP knowledge over time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.gp.evolve import EvolveResult, GPConfig, evolve
from engine.gp.strategy import GPStrategy
from engine.validation.gate import (
    GateThresholds,
    ValidationReport,
    evaluate_matrix,
    returns_for_configs,
)
from engine.validation.ledger import TrialLedger


def discover_gp(
    bars,
    *,
    ppy: float = 365.0,
    cfg: GPConfig | None = None,
    seed: int = 0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    thresholds: GateThresholds | None = None,
    ledger: TrialLedger | None = None,
    family: str | None = None,
    hof_path: Path | None = None,
    **gate_kwargs,
) -> ValidationReport:
    # Load prior-run HOF to warm-start this run's population.
    warm_start = None
    if hof_path is not None and Path(hof_path).exists():
        try:
            warm_start = json.loads(Path(hof_path).read_text())
        except Exception:
            pass

    result: EvolveResult = evolve(bars, ppy=ppy, cfg=cfg, seed=seed, warm_start=warm_start)

    # Persist HOF so the next nightly run can continue from here.
    if hof_path is not None:
        Path(hof_path).parent.mkdir(parents=True, exist_ok=True)
        Path(hof_path).write_text(json.dumps(result.hall_of_fame))

    configs = [{"program": program} for program, _ in result.hall_of_fame]
    matrix = returns_for_configs(
        bars, GPStrategy, configs, fee_bps=fee_bps, slippage_bps=slippage_bps, ppy=ppy
    )

    fam = family or "gp"
    trials = result.n_evaluated
    if ledger is not None:
        trials = ledger.add(fam, result.n_evaluated)  # count ALL evolved programs, not just HOF

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
        n_trials=trials,
        **gate_kwargs,
    )
