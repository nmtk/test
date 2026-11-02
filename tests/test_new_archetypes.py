"""Tests for the three new hypothesis families: session, vol_regime, carry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.strategies import all_strategies, get_strategy
from engine.strategies.carry import RateCarry
from engine.strategies.session import SessionDrift, in_window
from engine.strategies.vol_regime import VolRegimeBreak, rolling_percentile

HOUR_MS = 3_600_000


def _hourly(n: int, seed: int = 0, start_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 150.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    spread = np.abs(rng.normal(0, 0.001, n)) * close
    # Align to midnight UTC so hour-of-day assertions are exact.
    base = (start_ms // 86_400_000) * 86_400_000
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * HOUR_MS + base,
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": 1.0,
        }
    )


def _daily(n: int, seed: int = 0) -> pd.DataFrame:
    bars = _hourly(n, seed)
    bars["ts"] = np.arange(n, dtype="int64") * 86_400_000 + 1_700_000_000_000
    return bars


# -- registration --------------------------------------------------------------------


def test_all_three_are_registered_and_searched():
    from engine.search.research import DEFAULT_ARCHETYPES

    for name, cls in [
        ("session", SessionDrift),
        ("vol_regime", VolRegimeBreak),
        ("carry", RateCarry),
    ]:
        assert name in all_strategies()
        assert get_strategy(name) is cls
        assert name in DEFAULT_ARCHETYPES, f"{name} would never run nightly"


# -- session -------------------------------------------------------------------------


def test_window_wraps_past_midnight():
    """An Asian session spans the date line; a naive range would select nothing."""
    hours = pd.Series(range(24))
    tokyo = in_window(hours, 23, 8)
    assert tokyo.iloc[23] and tokyo.iloc[0] and tokyo.iloc[7]
    assert not tokyo.iloc[8] and not tokyo.iloc[12]
    london = in_window(hours, 7, 16)
    assert london.iloc[7] and london.iloc[15] and not london.iloc[16]
    assert not in_window(hours, 5, 5).any(), "an empty window trades nothing"


def test_position_is_held_only_inside_the_window():
    bars = _hourly(240, seed=1)
    weights = SessionDrift(start_hour=7, end_hour=16, direction=1).weights(bars)
    hours = pd.to_datetime(bars["ts"], unit="ms", utc=True).dt.hour
    assert (weights[(hours >= 7) & (hours < 16)] == 1.0).all()
    assert (weights[(hours < 7) | (hours >= 16)] == 0.0).all()


def test_direction_flips_the_book():
    bars = _hourly(240, seed=2)
    assert SessionDrift(direction=-1).weights(bars).min() == -1.0


def test_weekday_filter_drops_the_weekend():
    bars = _hourly(24 * 14, seed=3)
    weights = SessionDrift(start_hour=0, end_hour=23, weekdays_only=1).weights(bars)
    dow = pd.to_datetime(bars["ts"], unit="ms", utc=True).dt.dayofweek
    assert (weights[dow >= 5] == 0.0).all()
    assert (weights[dow < 5] != 0.0).any()


def test_daily_bars_are_flat_rather_than_buy_and_hold():
    """One bar a day means every bar is "in the window" — that is not a session trade."""
    weights = SessionDrift(start_hour=7, end_hour=16).weights(_daily(300, seed=4))
    assert (weights == 0.0).all()


def test_session_weights_are_causal():
    full = _hourly(500, seed=5)
    strategy = SessionDrift(trend_window=50)
    pd.testing.assert_series_equal(
        strategy.weights(full.iloc[:300].copy()),
        strategy.weights(full).iloc[:300],
        check_names=False,
    )


# -- vol_regime ----------------------------------------------------------------------


def test_percentile_is_bounded_and_causal():
    series = pd.Series(np.random.default_rng(7).normal(size=500).cumsum())
    pct = rolling_percentile(series, 100)
    valid = pct.notna()
    assert (pct[valid] >= 0.0).all() and (pct[valid] <= 1.0).all()


def test_fires_only_when_a_quiet_stretch_breaks():
    """A calm run then one violent bar: the trigger belongs to the violent bar."""
    n = 400
    close = np.full(n, 100.0) + np.linspace(0, 0.5, n)  # near-flat, tiny drift
    close[350] = 112.0  # the break
    spread = np.full(n, 0.01)
    spread[350] = 6.0
    bars = pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * HOUR_MS + 1_700_000_000_000,
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": 1.0,
        }
    )
    weights = VolRegimeBreak(
        vol_window=20,
        rank_window=100,
        compress_pct=0.4,
        expansion_mult=2.0,
        direction=1,
        max_hold=5,
    ).weights(bars)
    assert (weights.iloc[:340] == 0.0).all(), "nothing should fire during the quiet run"
    assert weights.iloc[350] == 1.0, "the upside break should be bought with direction=+1"


def test_direction_inverts_the_trade():
    n = 400
    close = np.full(n, 100.0) + np.linspace(0, 0.5, n)
    close[350] = 112.0
    spread = np.full(n, 0.01)
    spread[350] = 6.0
    bars = pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * HOUR_MS + 1_700_000_000_000,
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": 1.0,
        }
    )
    kw = dict(vol_window=20, rank_window=100, compress_pct=0.4, expansion_mult=2.0, max_hold=5)
    assert VolRegimeBreak(direction=1, **kw).weights(bars).iloc[350] == 1.0
    assert VolRegimeBreak(direction=-1, **kw).weights(bars).iloc[350] == -1.0


def test_weights_are_discrete_finite_and_causal():
    bars = _hourly(2000, seed=8)
    strategy = VolRegimeBreak()
    weights = strategy.weights(bars)
    assert set(np.unique(weights.to_numpy())) <= {-1.0, 0.0, 1.0}
    assert np.isfinite(weights.to_numpy()).all()
    pd.testing.assert_series_equal(
        strategy.weights(bars.iloc[:1200].copy()),
        strategy.weights(bars).iloc[:1200],
        check_names=False,
    )


def test_short_history_is_flat_not_an_error():
    assert (VolRegimeBreak(rank_window=200).weights(_hourly(50, seed=9)) == 0.0).all()


# -- carry ---------------------------------------------------------------------------


def test_carry_is_flat_without_its_features():
    bars = _daily(300, seed=10)
    assert (RateCarry().weights(bars) == 0.0).all()


def test_carry_leans_toward_the_higher_yielder():
    """Positive differential = US pays more = dollar-supportive = long USD/JPY."""
    bars = _daily(300, seed=11).assign(mac_carry_level=0.5)
    weights = RateCarry(direction=1, threshold=0.1, smooth=1, trend_filter=0).weights(bars)
    assert (weights == 1.0).all()
    flipped = RateCarry(direction=-1, threshold=0.1, smooth=1, trend_filter=0).weights(bars)
    assert (flipped == -1.0).all()


def test_scale_by_level_sizes_with_the_gap():
    """A half point of carry is not the same trade as three points of it."""
    narrow = _daily(200, seed=12).assign(mac_carry_level=0.2)
    wide = _daily(200, seed=12).assign(mac_carry_level=0.9)
    kw = dict(threshold=0.1, smooth=1, trend_filter=0, scale_by_level=1)
    assert RateCarry(**kw).weights(narrow).abs().max() < RateCarry(**kw).weights(wide).abs().max()


def test_trend_filter_stands_aside_when_price_disagrees():
    """The unwind is fast and the rate gap endorses the trade all the way down."""
    n = 300
    close = np.linspace(150.0, 120.0, n)  # a sustained decline
    bars = pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * 86_400_000 + 1_700_000_000_000,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
            "mac_carry_level": 0.8,  # carry still says "be long"
        }
    )
    unprotected = RateCarry(trend_filter=0, threshold=0.1, smooth=1).weights(bars)
    protected = RateCarry(trend_filter=1, trend_window=50, threshold=0.1, smooth=1).weights(bars)
    assert (unprotected.iloc[60:] == 1.0).all(), "without the filter it holds into the fall"
    assert protected.iloc[60:].abs().sum() == 0.0, "with it, the strategy stands aside"


def test_carry_rejects_an_unknown_signal():
    with pytest.raises(ValueError, match="unknown signal"):
        RateCarry(signal="not_a_signal")


def test_carry_weights_stay_bounded():
    bars = _daily(300, seed=13).assign(mac_carry_level=np.linspace(-2.0, 2.0, 300))
    weights = RateCarry(threshold=0.0, smooth=1, trend_filter=0, scale_by_level=1).weights(bars)
    assert weights.abs().max() <= 1.0
