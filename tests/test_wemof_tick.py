"""wemof_tick: WEMOF's entries with intrabar take-profit/stop exits, and its research path."""

from __future__ import annotations

import json

import numpy as np
import pytest

from engine.backtest import backtest_strategy
from engine.core.config import Settings, load_settings
from engine.data.dukascopy import ticks_to_ohlcv
from engine.data.intrabar import IntrabarPath, IntrabarStore, attach_intrabar, ticks_to_path
from engine.data.store import OHLCVStore
from engine.orchestration.population import Population
from engine.strategies import all_strategies, get_strategy
from engine.strategies.wemof import WemofReversal
from engine.strategies.wemof_tick import IntrabarRequired, WemofTick
from engine.validation.gate import GateThresholds, build_returns_matrix, evaluate_matrix
from engine.validation.ledger import TrialLedger

BAR = 300_000
LENIENT = GateThresholds(
    dsr_min=-1.0, pbo_max=1.0, spa_max=1.0, min_oos_sharpe=-1e9, min_pct_positive=0.0
)
VETOES_OFF = dict(veto_structure=0, veto_level=0, veto_news=0, veto_trend=0)


def _stamped(ticks):
    bars = ticks_to_ohlcv(ticks, "5m")
    path = IntrabarPath.from_frame(ticks_to_path(ticks, "USD/JPY"), "USD/JPY")
    return attach_intrabar(bars, path, BAR)


# -- shape ----------------------------------------------------------------------------------


def test_registered_with_the_original_exit_surface():
    assert get_strategy("wemof_tick") is WemofTick
    assert "wemof_tick" in all_strategies()
    space = WemofTick.param_space
    assert {"tp_pips", "sl_pips", "max_hold"} <= set(space)
    assert "stop_atr" not in space, "the ATR close-stop is the close-exit version's"
    assert WemofTick.requires_intrabar and WemofTick.intrabar_timeframes == ("5m",)
    assert WemofTick.parent_family == "wemof"


def test_it_refuses_to_pretend_it_is_a_weight_series(make_ticks):
    bars = ticks_to_ohlcv(make_ticks(hours=2), "5m")
    with pytest.raises(IntrabarRequired):
        WemofTick().weights(bars)
    with pytest.raises(IntrabarRequired, match="sync-data"):
        WemofTick().backtest(bars)


# -- behaviour --------------------------------------------------------------------------------


def test_trades_open_only_where_wemof_signals(make_ticks):
    bars = _stamped(make_ticks(hours=30, seed=4))
    params = dict(bb_window=20, bb_sigma=2.0, **VETOES_OFF)
    sim = WemofTick(**params, tp_pips=1.0, sl_pips=4.0, max_hold=3).simulate(bars)
    long_sig, short_sig, _ = WemofReversal(**params).entry_signals(bars)
    assert len(sim.trades) > 10
    for trade in sim.trades.itertuples():
        expected = long_sig if trade.side > 0 else short_sig
        assert expected[trade.signal_bar]


def test_levels_are_in_pips_from_the_fill(make_ticks):
    bars = _stamped(make_ticks(hours=30, seed=4))
    sim = WemofTick(bb_sigma=2.0, tp_pips=1.5, sl_pips=4.0, **VETOES_OFF).simulate(bars)
    takes = sim.trades[sim.trades["reason"] == "take_profit"]
    assert len(takes) > 0
    moved = (takes["exit_px"] - takes["entry_px"]) * takes["side"]
    np.testing.assert_allclose(moved.to_numpy(), 0.015, atol=1e-9)


def test_the_generic_backtest_runs_the_intrabar_simulation(make_ticks):
    bars = _stamped(make_ticks(hours=20, seed=6))
    strategy = WemofTick(bb_sigma=2.0, **VETOES_OFF)
    via_dispatch = backtest_strategy(strategy, bars, slippage_bps=0.3).returns
    direct = strategy.simulate(bars, slippage_bps=0.3).result.returns
    np.testing.assert_array_equal(via_dispatch.to_numpy(), direct.to_numpy())
    assert via_dispatch.index.equals(bars.index)


def test_the_gate_rejects_it_on_a_random_walk(make_ticks):
    """The noise self-test, through the intrabar path: no edge in, no challenger out."""
    bars = _stamped(make_ticks(hours=60, seed=9))
    grid = {"bb_sigma": [2.0, 2.5], "tp_pips": [1.0, 2.0], "sl_pips": [3.0, 6.0], "veto_news": [0]}
    matrix, configs = build_returns_matrix(bars, WemofTick, grid, slippage_bps=0.3, ppy=74_880)
    report = evaluate_matrix(
        matrix, name="wemof_tick:USD/JPY:5m", ppy=74_880, n_trials=len(configs)
    )
    assert matrix.shape == (len(bars), len(configs))
    assert np.isfinite(matrix).all()
    assert report.passed is False


# -- research -----------------------------------------------------------------------------------


@pytest.fixture
def fx_settings(tmp_path, make_ticks):
    base = load_settings()
    settings = Settings(
        repo_root=base.repo_root,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        config_dir=base.config_dir,
        live_enabled=False,
        alpaca_paper=True,
        bitbank=base.bitbank,
        alpaca=base.alpaca,
    )
    ticks = make_ticks(hours=30, seed=2)
    OHLCVStore(settings.data_dir / "ohlcv").upsert(
        "dukascopy", "USD/JPY", "5m", ticks_to_ohlcv(ticks, "5m")
    )
    ledger = TrialLedger.load(settings.state_dir / "trial_ledger.json")
    ledger.add("wemof:USD/JPY:5m", 1480)
    ledger.save()
    return settings, ticks


def _research(settings, monkeypatch):
    import engine.search.research as research_mod

    calls: list[dict] = []
    real = research_mod.discover_challenger

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(research_mod, "discover_challenger", spy)
    research_mod.run_research(
        settings,
        strategies=("wemof_tick",),
        symbols=["USD/JPY"],
        timeframes=["5m"],
        trials=3,
        thresholds=LENIENT,
        min_bars=150,
        include_ensemble=False,
    )
    return calls


def _ledger(settings) -> dict:
    return json.loads((settings.state_dir / "trial_ledger.json").read_text())


def test_research_waits_for_a_path_without_spending_trials(fx_settings, monkeypatch):
    settings, _ticks = fx_settings
    assert _research(settings, monkeypatch) == []
    assert "wemof_tick:USD/JPY:5m" not in _ledger(settings)


def test_research_waits_while_the_backfill_is_partial(fx_settings, monkeypatch):
    settings, ticks = fx_settings
    IntrabarStore(settings.data_dir / "intrabar").upsert(
        "dukascopy", "USD/JPY", ticks_to_path(ticks.iloc[: len(ticks) // 2], "USD/JPY")
    )
    assert _research(settings, monkeypatch) == []
    assert "wemof_tick:USD/JPY:5m" not in _ledger(settings)


def test_research_judges_it_research_only_at_venue_costs_with_inherited_trials(
    fx_settings, monkeypatch
):
    settings, ticks = fx_settings
    IntrabarStore(settings.data_dir / "intrabar").upsert(
        "dukascopy", "USD/JPY", ticks_to_path(ticks, "USD/JPY")
    )
    (call,) = _research(settings, monkeypatch)
    assert (call["fee_bps"], call["slippage_bps"]) == (0.0, 0.3)
    assert call["inherited_trials"] == 1480, "it re-searches wemof's entry space"

    pop = Population(settings.state_dir / "population.db")
    record = pop.get("wemof_tick:USD/JPY:5m")
    pop.close()
    assert record is not None
    assert record["metrics"]["research_only"] is True, "no executor can place its brackets"
    assert record["metrics"]["slippage_bps"] == 0.3
    assert _ledger(settings)["wemof_tick:USD/JPY:5m"] > 0


def test_research_skips_timeframes_the_method_does_not_use(fx_settings, monkeypatch):
    import engine.search.research as research_mod

    settings, ticks = fx_settings
    store = OHLCVStore(settings.data_dir / "ohlcv")
    store.upsert("dukascopy", "USD/JPY", "1h", ticks_to_ohlcv(ticks, "1h"))
    IntrabarStore(settings.data_dir / "intrabar").upsert(
        "dukascopy", "USD/JPY", ticks_to_path(ticks, "USD/JPY")
    )
    calls: list = []
    monkeypatch.setattr(research_mod, "discover_challenger", lambda *a, **k: calls.append(k))
    research_mod.run_research(
        settings,
        strategies=("wemof_tick",),
        symbols=["USD/JPY"],
        timeframes=["1h"],
        trials=3,
        min_bars=10,
        include_ensemble=False,
    )
    assert calls == []


def test_the_alpha_pool_never_takes_it_as_a_member(tmp_path):
    from engine.orchestration.population import StrategyRecord
    from engine.search.ensemble import gather_members

    pop = Population(tmp_path / "population.db")
    for name, params in (("wemof", {}), ("wemof_tick", {})):
        pop.upsert(
            StrategyRecord(
                family=f"{name}:USD/JPY:5m",
                strategy=name,
                symbol="USD/JPY",
                source="dukascopy",
                timeframe="5m",
                params=params,
                metrics={},
                status="candidate",
            )
        )
    members = gather_members(
        pop, symbol="USD/JPY", timeframe="5m", archetypes=("wemof", "wemof_tick")
    )
    pop.close()
    assert [m["strategy"] for m in members] == ["wemof"]
