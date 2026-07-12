"""Alpha-pool ensemble: member combination, decorrelation, discovery, and gate wiring."""

import json

import numpy as np
import pandas as pd

from engine.orchestration.population import Population, StrategyRecord
from engine.search.ensemble import discover_ensemble, gather_members
from engine.strategies import get_strategy
from engine.strategies.base import Strategy, register
from engine.strategies.ensemble import EnsembleStrategy
from engine.validation.ledger import TrialLedger

HOUR_MS = 3_600_000


def _bars(n=900, seed=0, drift=0.001, vol=0.005):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, n))
    return pd.DataFrame(
        {
            "ts": 1_600_000_000_000 + np.arange(n) * HOUR_MS,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
        }
    )


# Deterministic pool members: always-long harvests the drift; the even-bars variant is a
# genuinely distinct (partially correlated) bet; the clone duplicates always-long exactly.
@register
class _PoolAlwaysLong(Strategy):
    name = "_pool_always_long"

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)


@register
class _PoolEvenBars(Strategy):
    name = "_pool_even_bars"

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series((np.arange(len(bars)) % 2 == 0).astype(float), index=bars.index)


@register
class _PoolClone(Strategy):
    name = "_pool_clone"

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)


def _seed_population(path, entries):
    """entries: list of (family, strategy_name). Metrics/params minimal but valid."""
    pop = Population(path)
    for family, strategy_name in entries:
        name = family.split(":")[0]
        symbol, tf = family.split(":")[1], family.split(":")[2]
        assert name  # family format sanity
        pop.upsert(
            StrategyRecord(
                family=family,
                strategy=strategy_name,
                symbol=symbol,
                source="bitbank",
                timeframe=tf,
                params={},
                status="candidate",
                metrics={"cpcv_median": 1.0},
            )
        )
    return pop


# -- the strategy object ------------------------------------------------------------------


def test_ensemble_is_registered():
    assert get_strategy("ensemble") is EnsembleStrategy


def test_ensemble_weights_are_weighted_sum_clipped():
    bars = _bars(n=50)
    members = [
        {"strategy": "_pool_always_long", "params": {}, "weight": 0.8},
        {"strategy": "_pool_even_bars", "params": {}, "weight": 0.8},
    ]
    ens = EnsembleStrategy(members=members, k=2, weighting="equal")
    w = ens.weights(bars)
    # even bars: 0.8 + 0.8 = 1.6 -> clipped to 1.0; odd bars: 0.8 alone.
    assert (w.iloc[0::2] == 1.0).all()
    assert np.allclose(w.iloc[1::2], 0.8)
    assert w.abs().max() <= 1.0


def test_ensemble_params_survive_json_roundtrip():
    bars = _bars(n=60)
    members = [
        {"strategy": "_pool_always_long", "params": {}, "weight": 0.5},
        {"strategy": "_pool_even_bars", "params": {}, "weight": 0.5},
    ]
    original = EnsembleStrategy(members=members, k=2, weighting="inv_vol")
    rebuilt = get_strategy("ensemble")(**json.loads(json.dumps(original.params)))
    pd.testing.assert_series_equal(original.weights(bars), rebuilt.weights(bars))
    assert rebuilt.weighting == "inv_vol"


# -- member gathering ---------------------------------------------------------------------


def test_gather_members_reads_population_and_hof(tmp_path):
    pop = _seed_population(
        tmp_path / "population.db",
        [("tsmom:TST/JPY:1h", "_pool_always_long"), ("ma_cross:TST/JPY:1h", "_pool_even_bars")],
    )
    hof_path = tmp_path / "hof.json"
    hof_path.write_text(json.dumps([[["feat", "ret"], 1.0], [["feat", "close"], 0.5]]))

    members = gather_members(
        pop,
        symbol="TST/JPY",
        timeframe="1h",
        archetypes=("tsmom", "ma_cross", "donchian"),  # donchian row absent -> skipped
        hof_path=hof_path,
    )
    pop.close()
    assert len(members) == 4
    families = {m["family"] for m in members}
    assert families == {"tsmom:TST/JPY:1h", "ma_cross:TST/JPY:1h", "gp:TST/JPY:1h"}
    gp_members = [m for m in members if m["strategy"] == "gp"]
    assert gp_members[0]["params"] == {"program": ["feat", "ret"]}


def test_gather_members_skips_retired(tmp_path):
    pop = _seed_population(tmp_path / "population.db", [("tsmom:TST/JPY:1h", "_pool_always_long")])
    pop.update_fields("tsmom:TST/JPY:1h", status="retired")
    members = gather_members(
        pop, symbol="TST/JPY", timeframe="1h", archetypes=("tsmom",), hof_path=None
    )
    pop.close()
    assert members == []


# -- discovery + gate ---------------------------------------------------------------------


def test_discover_ensemble_forms_pool_and_inherits_trials(tmp_path):
    bars = _bars()
    pop = _seed_population(
        tmp_path / "population.db",
        [("tsmom:TST/JPY:1h", "_pool_always_long"), ("ma_cross:TST/JPY:1h", "_pool_even_bars")],
    )
    ledger = TrialLedger.load(tmp_path / "ledger.json")
    ledger.add("tsmom:TST/JPY:1h", 5000)  # the member's accumulated mining bill

    report = discover_ensemble(
        bars,
        population=pop,
        symbol="TST/JPY",
        timeframe="1h",
        archetypes=("tsmom", "ma_cross"),
        ppy=8760.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        ledger=ledger,
        family="ensemble:TST/JPY:1h",
    )
    pop.close()
    assert report is not None
    # k=2 x (equal, inv_vol) -> at least two structurally distinct pool variants
    assert report.n_configs >= 2
    members = report.best_params["members"]
    assert len(members) == 2
    assert abs(sum(m["weight"] for m in members) - 1.0) < 1e-6
    # DSR deflation inherits the deepest member family's cumulative count
    assert report.n_trials >= 5000 + report.n_configs
    assert ledger.get("ensemble:TST/JPY:1h") == report.n_configs


def test_discover_ensemble_dedupes_clones_to_none(tmp_path):
    bars = _bars()
    # Two members with byte-identical position streams are ONE bet: after the correlation
    # dedupe a single member remains, and a pool of one must not be re-gated as new.
    pop = _seed_population(
        tmp_path / "population.db",
        [("tsmom:TST/JPY:1h", "_pool_always_long"), ("ma_cross:TST/JPY:1h", "_pool_clone")],
    )
    report = discover_ensemble(
        bars,
        population=pop,
        symbol="TST/JPY",
        timeframe="1h",
        archetypes=("tsmom", "ma_cross"),
        ppy=8760.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    pop.close()
    assert report is None


def test_discover_ensemble_drops_unprofitable_members(tmp_path):
    # A downtrend makes always-long lose -> CPCV median <= 0 -> no members -> no pool.
    bars = _bars(drift=-0.001)
    pop = _seed_population(
        tmp_path / "population.db",
        [("tsmom:TST/JPY:1h", "_pool_always_long"), ("ma_cross:TST/JPY:1h", "_pool_even_bars")],
    )
    report = discover_ensemble(
        bars,
        population=pop,
        symbol="TST/JPY",
        timeframe="1h",
        archetypes=("tsmom", "ma_cross"),
        ppy=8760.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    pop.close()
    assert report is None
