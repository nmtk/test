"""The genetic-programming loop: evolve a population of alpha programs.

Tournament selection + subtree crossover/mutation + elitism, scored by :func:`fitness`.
Deterministic given a seed. Returns a hall-of-fame of distinct programs (the candidate
*family* the validation gate then judges) plus the total number of distinct programs
evaluated (so the trial ledger — and thus the Deflated Sharpe Ratio — counts GP honestly).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.gp.features import build_features
from engine.gp.fitness import fitness
from engine.gp.program import (
    Node,
    clone,
    crossover,
    deserialize,
    mutate,
    random_tree,
    serialize,
)


@dataclass(frozen=True, slots=True)
class GPConfig:
    pop_size: int = 60
    generations: int = 8
    max_depth: int = 4
    tournament: int = 3
    p_crossover: float = 0.7
    p_mutation: float = 0.2
    elitism: int = 2
    hall_of_fame: int = 12
    complexity_penalty: float = 0.002
    turnover_penalty: float = 0.25
    cpcv_groups: int = 6
    cpcv_test_groups: int = 2
    fee_bps: float = 10.0
    slippage_bps: float = 5.0


@dataclass(slots=True)
class EvolveResult:
    hall_of_fame: list[tuple[list, float]]  # (serialized program, fitness)
    n_evaluated: int  # distinct programs evaluated (for honest DSR deflation)


def _tournament(scored: list[tuple[Node, float]], rng: np.random.Generator, k: int) -> Node:
    idx = rng.integers(0, len(scored), size=k)
    best = max((scored[i] for i in idx), key=lambda x: x[1])
    return clone(best[0])


def evolve(
    bars: pd.DataFrame,
    *,
    ppy: float = 365.0,
    cfg: GPConfig | None = None,
    seed: int = 0,
    warm_start: list[tuple[list, float]] | None = None,
) -> EvolveResult:
    cfg = cfg or GPConfig()
    rng = np.random.default_rng(seed)
    features = build_features(bars)
    cache: dict[str, float] = {}

    def fit(node: Node) -> float:
        key = json.dumps(serialize(node))
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = fitness(
            node,
            bars,
            features,
            ppy=ppy,
            fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps,
            complexity_penalty=cfg.complexity_penalty,
            turnover_penalty=cfg.turnover_penalty,
            cpcv_groups=cfg.cpcv_groups,
            cpcv_test_groups=cfg.cpcv_test_groups,
        )
        cache[key] = value
        return value

    if warm_start:
        # Seed the population with survivors from the previous run, then fill randomly.
        # Silently drop any programs that fail to deserialize (e.g. after operator changes).
        warm_nodes: list[Node] = []
        for prog, _ in warm_start:
            try:
                warm_nodes.append(deserialize(prog))
            except Exception:
                pass
        n_warm = min(len(warm_nodes), cfg.pop_size)
        population = warm_nodes[:n_warm] + [
            random_tree(rng, max_depth=cfg.max_depth) for _ in range(cfg.pop_size - n_warm)
        ]
    else:
        population = [random_tree(rng, max_depth=cfg.max_depth) for _ in range(cfg.pop_size)]
    scored = [(p, fit(p)) for p in population]

    for _ in range(cfg.generations):
        scored.sort(key=lambda x: x[1], reverse=True)
        next_pop: list[Node] = [clone(p) for p, _ in scored[: cfg.elitism]]
        while len(next_pop) < cfg.pop_size:
            r = rng.random()
            parent = _tournament(scored, rng, cfg.tournament)
            if r < cfg.p_crossover:
                child = crossover(parent, _tournament(scored, rng, cfg.tournament), rng)
            elif r < cfg.p_crossover + cfg.p_mutation:
                child = mutate(parent, rng, max_depth=cfg.max_depth)
            else:
                child = parent
            next_pop.append(child)
        scored = [(p, fit(p)) for p in next_pop]

    scored.sort(key=lambda x: x[1], reverse=True)
    hof: list[tuple[list, float]] = []
    seen: set[str] = set()
    for node, value in scored:
        key = json.dumps(serialize(node))
        if key in seen:
            continue
        seen.add(key)
        hof.append((serialize(node), value))
        if len(hof) >= cfg.hall_of_fame:
            break
    return EvolveResult(hall_of_fame=hof, n_evaluated=len(cache))
