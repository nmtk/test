"""The gate must score the exposure the venue allows.

``engine.orchestration.risk.apply_limits`` clamps a negative target weight to zero
whenever ``can_short`` is false, and bitbank spot crypto is long-only. That clamp lived
only in the executors: the search, the validation gate, the forward test and the allocator
all scored unconstrained long/short weights. So the gate's verdict was about a strategy no
order could ever take.

It was not hypothetical. ``ensemble:ETH/JPY:1d`` cleared the full gate (Sharpe 2.89,
PBO 0.18, SPA p 0.000) and was the one row of twelve whose forward test looked like
passing — +0.2982 per bar against a required +0.1162. It then turned out to be **short on
100% of its forward window**. Long-only it never holds anything at all, so its achievable
forward Sharpe is +0.0000 and it fails. Two more rows were 78% and 61% short. With the
constraint applied, the passing count went from 1 of 12 to 0 of 12.

This is the third instance of one shape of bug in this engine: the gate measuring one
thing and the executor doing another (the others were the missing context augmentation and
the bar-count forward window). Hence the parameters that carry the constraint are required
rather than defaulted, at every pipeline entry point.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from engine.backtest import backtest_strategy
from engine.core.types import AssetClass, Instrument, Venue
from engine.orchestration.risk import can_short


class _AlwaysShort:
    def weights(self, bars):
        return pd.Series(-1.0, index=bars.index)


class _AlwaysLong:
    def weights(self, bars):
        return pd.Series(1.0, index=bars.index)


class _Alternating:
    def weights(self, bars):
        return pd.Series(np.where(np.arange(len(bars)) % 2 == 0, 1.0, -1.0), index=bars.index)


def _bars(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * 86_400_000,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


# -- the venue rule itself -------------------------------------------------------------


def test_bitbank_spot_cannot_be_shorted_but_fx_can():
    spot = Instrument(
        symbol="ETH/JPY",
        asset_class=AssetClass.CRYPTO,
        venue=Venue.BITBANK,
        base="ETH",
        quote="JPY",
    )
    fx = Instrument(
        symbol="USD/JPY", asset_class=AssetClass.FX, venue=Venue.OANDA, base="USD", quote="JPY"
    )
    assert can_short(spot) is False
    assert can_short(fx) is True


# -- what the backtester measures ------------------------------------------------------


def test_a_short_only_strategy_measures_as_flat_on_a_long_only_venue():
    """The exact shape of ensemble:ETH/JPY:1d's forward window."""
    bars = _bars()
    free = backtest_strategy(_AlwaysShort(), bars, long_only=False)
    held = backtest_strategy(_AlwaysShort(), bars, long_only=True)
    assert np.count_nonzero(free.returns.to_numpy()) > 0, "unconstrained, it trades"
    assert np.count_nonzero(held.returns.to_numpy()) == 0, (
        "long-only, the executor would hold nothing — so neither may the measurement"
    )


def test_a_long_only_strategy_is_unaffected():
    bars = _bars()
    a = backtest_strategy(_AlwaysLong(), bars, long_only=False)
    b = backtest_strategy(_AlwaysLong(), bars, long_only=True)
    np.testing.assert_allclose(a.returns.to_numpy(), b.returns.to_numpy())


def test_only_the_short_legs_are_dropped():
    bars = _bars()
    held = backtest_strategy(_Alternating(), bars, long_only=True)
    applied = held.applied_weights.to_numpy()
    assert (applied >= 0).all(), "no negative exposure may survive"
    assert np.count_nonzero(applied) > 0, "the long legs must still be taken"


# -- the pipeline must decide, not default ---------------------------------------------


@pytest.mark.parametrize(
    "dotted",
    [
        "engine.search.optuna_search.discover_challenger",
        "engine.search.gp_search.discover_gp",
        "engine.search.ensemble.discover_ensemble",
        "engine.orchestration.promotion.forward_returns",
        "engine.orchestration.promotion._strategy_returns",
    ],
)
def test_long_only_is_required_at_every_entry_point(dotted):
    """A default is what let the clamp go missing for months; a caller must choose."""
    module_name, _, attr = dotted.rpartition(".")
    module = __import__(module_name, fromlist=[attr])
    params = inspect.signature(getattr(module, attr)).parameters
    assert "long_only" in params, f"{dotted} does not take the venue constraint"
    assert params["long_only"].default is inspect.Parameter.empty, (
        f"{dotted}'s long_only has a default, so a caller can silently omit it"
    )


def test_research_derives_the_constraint_from_the_universe():
    """It must come from the instrument, not from a hand-maintained symbol list."""
    import engine.search.research as research

    source = inspect.getsource(research)
    assert "can_short(" in source, "research no longer consults the venue rule"
    assert "long_only=entry.long_only" in source


def test_promotion_derives_it_per_record():
    from engine.core.config import load_settings, load_universe
    from engine.orchestration.promotion import venue_long_only

    universe = load_universe(load_settings().universe_path)
    assert venue_long_only({"symbol": "ETH/JPY"}, universe) is True
    assert venue_long_only({"symbol": "USD/JPY"}, universe) is False
    # Not in the universe at all: no venue, so no constraint (and barred from promotion).
    assert venue_long_only({"symbol": "DEXJPUS"}, universe) is False
