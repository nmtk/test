"""FX wiring: asset class, cost assumptions, universe overrides, and the research path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.backtest.metrics import periods_per_year
from engine.core.config import load_settings, load_universe
from engine.core.types import AssetClass, Venue
from engine.data.sync import DERIVE_SOURCES
from engine.execution.broker import RESEARCH_COSTS_BPS, research_costs

# -- annualization -----------------------------------------------------------------------


def test_fx_annualizes_on_a_five_day_week():
    """Spot FX runs 24h Sunday evening to Friday evening: ~260 days, not 365 or 252x6.5h."""
    assert periods_per_year("1d", AssetClass.FX) == pytest.approx(260.0)
    assert periods_per_year("1h", AssetClass.FX) == pytest.approx(260 * 24)
    assert periods_per_year("5m", AssetClass.FX) == pytest.approx(260 * 24 * 12)


def test_fx_sits_between_crypto_and_equity_bar_counts():
    fx = periods_per_year("1d", AssetClass.FX)
    assert (
        periods_per_year("1d", AssetClass.EQUITY) < fx < periods_per_year("1d", AssetClass.CRYPTO)
    )


# -- cost assumptions --------------------------------------------------------------------


def test_fx_research_costs_are_an_order_of_magnitude_below_crypto():
    """USD/JPY pays a ~0.3bps spread; charging it crypto's 15bps rejects real edges."""
    fx_fee, fx_slip = research_costs(AssetClass.FX)
    crypto_fee, crypto_slip = research_costs(AssetClass.CRYPTO)
    assert fx_fee + fx_slip == pytest.approx(1.0)
    assert crypto_fee + crypto_slip == pytest.approx(15.0)
    assert (fx_fee + fx_slip) * 10 <= (crypto_fee + crypto_slip)


def test_every_asset_class_has_a_cost_assumption():
    for asset_class in AssetClass:
        assert asset_class in RESEARCH_COSTS_BPS, f"{asset_class} has no research cost"


def test_unknown_asset_class_falls_back_to_the_conservative_figure():
    assert research_costs("not-an-asset-class") == RESEARCH_COSTS_BPS[AssetClass.CRYPTO]


# -- universe ----------------------------------------------------------------------------


def test_usdjpy_is_configured_for_fx_execution_with_a_keyless_feed():
    universe = load_universe(load_settings().universe_path)
    entry = next(e for e in universe.entries if e.instrument.symbol == "USD/JPY")

    assert entry.instrument.asset_class is AssetClass.FX
    assert entry.instrument.venue is Venue.OANDA
    # The data override is the point: bars sync with no broker credentials at all.
    assert entry.live_source.source == "dukascopy"
    # 5m is what makes the gate's minimum track record reachable (see docs/STRATEGY.md).
    assert "5m" in entry.timeframes
    # v20 takes integer units of the base currency.
    assert entry.instrument.size_step == 1.0


def test_data_override_is_optional():
    """Instruments without a `data:` block still follow their venue's default feed."""
    universe = load_universe(load_settings().universe_path)
    btc = next(e for e in universe.entries if e.instrument.symbol == "BTC/JPY")
    assert btc.live_source.source == "bitbank"


def test_higher_fx_timeframes_are_derived_not_refetched():
    """One file per instrument-hour: fetching 5m/1h/1d natively would triple the budget."""
    assert "dukascopy" in DERIVE_SOURCES


# -- the research path, end to end -------------------------------------------------------


def _synthetic_usdjpy_5m(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 150.0 * np.exp(np.cumsum(rng.normal(0, 0.0004, n)))
    spread = np.abs(rng.normal(0, 0.0003, n)) * close
    step_ms = 5 * 60 * 1000
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * step_ms + 1_700_000_000_000,
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(50, 500, n).astype(float),
        }
    )


def test_wemof_runs_through_the_gate_on_fx_bars(tmp_path):
    """The whole path: FX bars -> augmenter -> archetype -> validation gate."""
    from engine.data.store import OHLCVStore
    from engine.features.context import ContextAugmenter
    from engine.strategies.wemof import WemofReversal
    from engine.validation.gate import build_returns_matrix, evaluate_matrix

    bars = _synthetic_usdjpy_5m()
    store = OHLCVStore(tmp_path / "ohlcv")
    store.upsert("dukascopy", "USD/JPY", "5m", bars)

    settings = load_settings()
    universe = load_universe(settings.universe_path)
    augmenter = ContextAugmenter(store=store, series=[], market_by_source={})
    stamped = augmenter.augment(
        store.read("dukascopy", "USD/JPY", "5m"),
        source="dukascopy",
        symbol="USD/JPY",
        timeframe="5m",
    )
    # The macro layers must stamp neutral columns rather than vanish on an FX frame.
    assert "ctx_risk_on" in stamped.columns
    assert "mac_balance_diff" in stamped.columns

    entry = next(e for e in universe.entries if e.instrument.symbol == "USD/JPY")
    ppy = periods_per_year("5m", entry.instrument.asset_class)
    fee, slip = research_costs(entry.instrument.asset_class)
    grid = {"bb_window": [20], "bb_sigma": [2.0, 3.0], "veto_news": [0, 1]}
    matrix, configs = build_returns_matrix(
        stamped, WemofReversal, grid, fee_bps=fee, slippage_bps=slip, ppy=ppy
    )
    report = evaluate_matrix(matrix, name="wemof:USD/JPY:5m", ppy=ppy, n_trials=len(configs))

    assert matrix.shape == (len(stamped), 4)
    assert np.isfinite(matrix).all()
    # Random-walk input has no edge, so the gate must reject it. This is the noise
    # self-test applied to the FX path.
    assert report.passed is False
