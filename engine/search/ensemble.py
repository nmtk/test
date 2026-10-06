"""Alpha-pool discovery: turn the population's near-misses into one gate-worthy pool.

Motivation (see docs/STRATEGY.md): after ~126k trials, GP families reached DSR ≈ 1.0,
SPA p ≈ 0 and CPCV medians of +2..+3 yet every one failed the gate on PBO alone. CSCV's
PBO judges *argmax selection* — "pick the in-sample best, does it stay best?" — and among
a hall of fame of near-equal siblings that pick is a coin flip, so PBO sits near 0.5 no
matter how good the members are. The deployable answer is a **fixed-weight pool** of
decorrelated members: no per-sample argmax to destabilize, and diversification lifts the
pool's Sharpe above its members'. The pool family (a handful of structurally different
breadth/weighting variants) goes through the **unchanged** gate.

Honest accounting: the members were data-mined, so the pool's DSR is deflated by the
pool family's own cumulative trials **plus the largest member family's cumulative count**
(``n_trials = pool + max(member families)``). That inheritance is stricter than what any
other family faces; nothing about the thresholds changes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import engine.gp.strategy  # noqa: F401  -- registers "gp" so HOF members reconstruct
from engine.backtest import run_backtest
from engine.gp.evolve import max_abs_correlation
from engine.strategies import get_strategy
from engine.strategies.ensemble import EnsembleStrategy
from engine.validation.cpcv import cpcv_sharpes
from engine.validation.gate import (
    GateThresholds,
    ValidationReport,
    evaluate_matrix,
    returns_for_configs,
)
from engine.validation.ledger import TrialLedger

DEFAULT_KS = (2, 3, 5, 8)
DEFAULT_WEIGHTINGS = ("equal", "inv_vol")
# Members whose net-return streams correlate above this are one bet, not two; keep the
# higher-scoring one so pool breadth is real diversification.
MEMBER_CORR_CAP = 0.9


def gather_members(
    population,
    *,
    symbol: str,
    timeframe: str,
    archetypes: tuple[str, ...],
    hof_path: Path | None = None,
    max_gp_members: int = 12,
) -> list[dict]:
    """Collect candidate pool members: each archetype's best params + the GP hall of fame.

    Every member carries its population ``family`` string so the caller can inherit the
    member families' trial counts into the pool's DSR deflation.
    """
    members: list[dict] = []
    for name in archetypes:
        if getattr(get_strategy(name), "requires_intrabar", False):
            # Fills inside the bar: it has no weight series to blend into a pool.
            continue
        family = f"{name}:{symbol}:{timeframe}"
        row = population.get(family)
        if not row or row["status"] == "retired":
            continue  # retired = decayed on forward data; don't recycle it into the pool
        members.append({"strategy": row["strategy"], "params": row["params"], "family": family})
    if hof_path is not None and Path(hof_path).exists():
        try:
            hof = json.loads(Path(hof_path).read_text())
        except Exception:
            hof = []
        gp_family = f"gp:{symbol}:{timeframe}"
        for program, _fitness in hof[:max_gp_members]:
            members.append({"strategy": "gp", "params": {"program": program}, "family": gp_family})
    return members


def _net_returns(bars, member: dict, *, fee_bps: float, slippage_bps: float, ppy: float):
    weights = get_strategy(member["strategy"])(**member["params"]).weights(bars)
    result = run_backtest(
        bars["close"], weights, fee_bps=fee_bps, slippage_bps=slippage_bps, periods_per_year=ppy
    )
    return result.returns


def discover_ensemble(
    bars,
    *,
    population,
    symbol: str,
    timeframe: str,
    archetypes: tuple[str, ...],
    hof_path: Path | None = None,
    ppy: float = 365.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    thresholds: GateThresholds | None = None,
    ledger: TrialLedger | None = None,
    family: str | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
    weightings: tuple[str, ...] = DEFAULT_WEIGHTINGS,
    corr_cap: float = MEMBER_CORR_CAP,
    min_member_score: float = 0.0,
    max_gp_members: int = 12,
    **gate_kwargs,
) -> ValidationReport | None:
    """Build pool variants from the population + GP HOF and run them through the gate.

    Returns ``None`` when fewer than two decorrelated profitable members exist — a pool
    of one is just that strategy again and must not be re-judged as something new.
    """
    candidates = gather_members(
        population,
        symbol=symbol,
        timeframe=timeframe,
        archetypes=archetypes,
        hof_path=hof_path,
        max_gp_members=max_gp_members,
    )

    # Score members by out-of-sample-style robustness (CPCV median of net returns) and
    # drop anything unprofitable, degenerate, or non-finite.
    scored: list[tuple[float, dict, np.ndarray]] = []
    for member in candidates:
        try:
            returns = _net_returns(
                bars, member, fee_bps=fee_bps, slippage_bps=slippage_bps, ppy=ppy
            )
        except Exception:
            continue
        arr = returns.to_numpy(dtype=float)
        if len(arr) == 0 or not np.isfinite(arr).all() or float(np.std(arr)) == 0.0:
            continue
        score = cpcv_sharpes(returns, ppy=ppy).median
        if not math.isfinite(score) or score <= min_member_score:
            continue
        scored.append((score, member, arr))
    scored.sort(key=lambda item: item[0], reverse=True)

    # Greedy decorrelation: best first, then only members adding a genuinely new bet.
    accepted: list[tuple[float, dict, np.ndarray]] = []
    for score, member, arr in scored:
        if max_abs_correlation(arr, [a for _, _, a in accepted]) > corr_cap:
            continue
        accepted.append((score, member, arr))
    if len(accepted) < 2:
        return None

    # Pool variants: breadth (top-K) x weighting rule, weights frozen into the params.
    configs: list[dict] = []
    for k in ks:
        if k > len(accepted):
            continue
        subset = accepted[:k]
        vols = np.array([float(np.std(arr)) for _, _, arr in subset])
        for weighting in weightings:
            if weighting == "equal":
                w = np.full(k, 1.0 / k)
            elif weighting == "inv_vol":
                inv = 1.0 / vols  # vols > 0 guaranteed by the degenerate filter above
                w = inv / inv.sum()
            else:
                raise ValueError(f"unknown weighting {weighting!r}")
            members = [
                {
                    "strategy": member["strategy"],
                    "params": member["params"],
                    "weight": round(float(wi), 6),
                }
                for (_, member, _), wi in zip(subset, w, strict=True)
            ]
            configs.append({"members": members, "k": k, "weighting": weighting})
    if len(configs) < 2:
        return None

    matrix = returns_for_configs(
        bars, EnsembleStrategy, configs, fee_bps=fee_bps, slippage_bps=slippage_bps, ppy=ppy
    )

    fam = family or f"ensemble:{symbol}:{timeframe}"
    # DSR deflation: the pool's own cumulative variants PLUS the deepest-mined member
    # family's cumulative count — the pool inherits its members' data-mining bill.
    member_families = {member["family"] for _, member, _ in accepted}
    inherited = max((ledger.get(f) for f in member_families), default=0) if ledger else 0
    trials = len(configs)
    if ledger is not None:
        trials = ledger.add(fam, len(configs))
    n_trials = trials + inherited

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
        n_trials=n_trials,
        **gate_kwargs,
    )
