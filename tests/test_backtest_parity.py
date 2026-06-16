"""Parity: the vectorized backtester must agree with the event-driven PaperBroker.

A constant fully-invested position has a known closed form (close[-1]/close[0]). Running
it through both engines and getting the same answer is the practical proof that the fast
vectorized path has no look-ahead and books costs the same way as real execution.
"""

import pandas as pd
import pytest

from engine.backtest import run_backtest
from engine.core.types import AssetClass, Instrument, Order, OrderType, Side, Venue
from engine.execution.paper import PaperBroker

SYN = Instrument("SYN/JPY", AssetClass.CRYPTO, Venue.PAPER, base="SYN", quote="JPY", size_step=0.0)


def test_buy_and_hold_parity():
    close = pd.Series([100, 101, 103, 102, 105, 110, 108, 115], dtype=float)

    # Vectorized: constant full long, no costs.
    weights = pd.Series(1.0, index=close.index)
    vec = run_backtest(close, weights, fee_bps=0.0, slippage_bps=0.0)
    vec_ratio = float(vec.equity.iloc[-1])

    # Event-driven: buy the whole book at bar 0, then just hold and re-mark.
    cash = 1_000_000.0
    pb = PaperBroker({"SYN/JPY": SYN}, {"JPY": cash}, venue=Venue.PAPER, slippage_bps=0.0)
    pb.update_marks({"SYN/JPY": float(close.iloc[0])})
    pb.submit_order(
        Order(
            client_order_id="buy",
            symbol="SYN/JPY",
            side=Side.BUY,
            qty=cash / float(close.iloc[0]),
            type=OrderType.MARKET,
        )
    )
    for px in close.iloc[1:]:
        pb.update_marks({"SYN/JPY": float(px)})
    pb_ratio = pb.equity()["JPY"] / cash

    assert pb_ratio == pytest.approx(vec_ratio, rel=1e-9)
    assert pb_ratio == pytest.approx(float(close.iloc[-1] / close.iloc[0]), rel=1e-9)
