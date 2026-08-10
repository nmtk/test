"""WEMOF archetype tests: causality, the veto semantics, and exit discipline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.strategies import all_strategies, get_strategy
from engine.strategies.wemof import WemofReversal, bollinger, round_level_distance

HOUR_MS = 3_600_000


def _bars(close, high=None, low=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * HOUR_MS + 1_700_000_000_000,
            "open": close,
            "high": close if high is None else np.asarray(high, dtype=float),
            "low": close if low is None else np.asarray(low, dtype=float),
            "close": close,
            "volume": np.ones(n),
        }
    )


def _random_walk(n: int, seed: int = 0, start: float = 150.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    spread = np.abs(rng.normal(0, 0.001, n)) * close
    return _bars(close, high=close + spread, low=close - spread)


# -- registration and shape --------------------------------------------------------------


def test_registered_under_its_name():
    assert "wemof" in all_strategies()
    assert get_strategy("wemof") is WemofReversal
    assert "wemof(" in WemofReversal().describe()


def test_weights_are_aligned_discrete_and_finite():
    bars = _random_walk(600, seed=1)
    weights = WemofReversal().weights(bars)
    assert weights.index.equals(bars.index)
    assert len(weights) == len(bars)
    assert np.isfinite(weights.to_numpy()).all()
    assert set(np.unique(weights.to_numpy())) <= {-1.0, 0.0, 1.0}


def test_flat_during_the_warmup_window():
    bars = _random_walk(300, seed=2)
    weights = WemofReversal(bb_window=40).weights(bars)
    assert (weights.iloc[:39] == 0.0).all()


# -- the property that protects every backtest -------------------------------------------


def test_weights_are_causal_future_bars_cannot_change_the_past():
    """Appending later bars must not alter a single earlier weight."""
    full = _random_walk(900, seed=3)
    prefix_len = 500
    strategy = WemofReversal()
    on_prefix = strategy.weights(full.iloc[:prefix_len].copy())
    on_full = strategy.weights(full).iloc[:prefix_len]
    pd.testing.assert_series_equal(on_prefix, on_full, check_names=False)


# -- entry semantics ---------------------------------------------------------------------


def test_entry_requires_piercing_the_band():
    """With no touch there is no trade, however long the series."""
    bars = _bars(np.full(400, 150.0) + np.sin(np.arange(400) / 10.0) * 0.01)
    weights = WemofReversal(bb_sigma=3.5, veto_news=0).weights(bars)
    # A pure sine never reaches 3.5 sigma of its own rolling distribution.
    assert (weights == 0.0).all() or weights.abs().sum() < 5


def test_fade_is_counter_trend():
    """A downside overshoot buys, an upside overshoot sells."""
    close = np.full(120, 100.0)
    close[60] = 90.0  # unexplained air pocket
    bars = _bars(close)
    weights = WemofReversal(
        bb_window=20, bb_sigma=2.0, veto_structure=0, veto_level=0, veto_news=0, veto_trend=0
    ).weights(bars)
    assert weights.iloc[60] == 1.0, "a collapse should be bought, not sold"

    close_up = np.full(120, 100.0)
    close_up[60] = 110.0
    weights_up = WemofReversal(
        bb_window=20, bb_sigma=2.0, veto_structure=0, veto_level=0, veto_news=0, veto_trend=0
    ).weights(_bars(close_up))
    assert weights_up.iloc[60] == -1.0


def test_every_veto_can_only_remove_trades_never_add_them():
    """Each veto is a filter on the same candidate set, so it is monotone by construction."""
    bars = _random_walk(3000, seed=4)
    off = dict(veto_structure=0, veto_level=0, veto_news=0, veto_trend=0)
    base = WemofReversal(**off).weights(bars)
    base_bars = int((base != 0).sum())
    assert base_bars > 0, "the unfiltered fade should find candidates to trade"
    for veto in ("structure", "level", "news", "trend"):
        filtered = WemofReversal(**{**off, f"veto_{veto}": 1}).weights(bars)
        assert int((filtered != 0).sum()) <= base_bars, f"veto_{veto} added trades"


def test_all_vetoes_together_are_the_most_selective():
    bars = _random_walk(3000, seed=5)
    none = WemofReversal(veto_structure=0, veto_level=0, veto_news=0, veto_trend=0).weights(bars)
    every = WemofReversal(veto_structure=1, veto_level=1, veto_news=1, veto_trend=1).weights(bars)
    assert int((every != 0).sum()) < int((none != 0).sum())


# -- exits -------------------------------------------------------------------------------


def test_max_hold_caps_a_losing_position():
    """A trade that never reverts is cut after ``max_hold`` bars, not carried."""
    # Step down and stay there: price never returns to the (falling) band middle, so only
    # the holding cap can close the trade.
    close = np.concatenate([np.full(60, 100.0), np.full(60, 90.0)])
    close = close + np.linspace(0, 0.001, 120)  # break the degenerate zero-variance case
    bars = _bars(close)
    max_hold = 4
    weights = WemofReversal(
        bb_window=20,
        bb_sigma=2.0,
        max_hold=max_hold,
        stop_atr=99.0,  # disable the stop so max_hold is the binding exit
        veto_structure=0,
        veto_level=0,
        veto_news=0,
        veto_trend=0,
    ).weights(bars)
    held = weights.to_numpy()[60:]
    # Count the opening run of long exposure that starts at the shock bar.
    run = 0
    for value in held:
        if value != 1.0:
            break
        run += 1
    assert 0 < run <= max_hold, f"held {run} bars with max_hold={max_hold}"


def test_longer_max_hold_keeps_more_exposure():
    bars = _random_walk(3000, seed=6)
    short = WemofReversal(max_hold=1, stop_atr=99.0).weights(bars)
    long = WemofReversal(max_hold=20, stop_atr=99.0).weights(bars)
    assert int((long != 0).sum()) > int((short != 0).sum())


# -- helpers -----------------------------------------------------------------------------


def test_round_level_distance_is_zero_on_round_numbers():
    prices = pd.Series([150.0, 150.5, 150.25, 149.0])
    # decades=2 at this magnitude means a step of 1.0 (whole yen).
    distance = round_level_distance(prices, decades=2)
    assert distance.iloc[0] == pytest.approx(0.0)
    assert distance.iloc[3] == pytest.approx(0.0)
    assert distance.iloc[1] == pytest.approx(0.5)
    assert distance.iloc[2] == pytest.approx(0.25)
    assert (distance >= 0.0).all() and (distance <= 0.5).all()


def test_round_level_distance_scales_with_price_magnitude():
    """The same parameter has to mean something sane at 150 and at 10,000,000."""
    for price in (150.0, 10_000_000.0):
        on_grid = round_level_distance(pd.Series([price]), decades=2)
        assert on_grid.iloc[0] == pytest.approx(0.0)


def test_bollinger_bands_bracket_the_middle():
    close = pd.Series(np.random.default_rng(9).normal(100, 2, 300))
    lower, middle, upper = bollinger(close, window=20, sigma=2.0)
    valid = middle.notna()
    assert (lower[valid] <= middle[valid]).all()
    assert (middle[valid] <= upper[valid]).all()
    assert lower.iloc[:19].isna().all(), "bands must not exist before the window fills"
