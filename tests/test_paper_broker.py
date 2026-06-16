import pytest

from engine.core.types import (
    AssetClass,
    Instrument,
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    Venue,
)
from engine.execution.paper import PaperBroker

BTC = Instrument(
    "BTC/JPY", AssetClass.CRYPTO, Venue.BITBANK, base="BTC", quote="JPY", size_step=0.0001
)
INSTRUMENTS = {"BTC/JPY": BTC}


def _mkt(coid, side, qty):
    return Order(client_order_id=coid, symbol="BTC/JPY", side=side, qty=qty, type=OrderType.MARKET)


def _limit(coid, side, qty, price):
    return Order(
        client_order_id=coid,
        symbol="BTC/JPY",
        side=side,
        qty=qty,
        type=OrderType.LIMIT,
        limit_price=price,
        tif=TimeInForce.POST_ONLY,
    )


def test_market_requires_mark():
    pb = PaperBroker(INSTRUMENTS, {"JPY": 1_000_000_000}, slippage_bps=0)
    with pytest.raises(ValueError):
        pb.submit_order(_mkt("a", Side.BUY, 0.1))


def test_position_average_and_realized_pnl():
    pb = PaperBroker(INSTRUMENTS, {"JPY": 1_000_000_000}, venue=Venue.PAPER, slippage_bps=0)
    pb.update_marks({"BTC/JPY": 100})
    pb.submit_order(_mkt("b1", Side.BUY, 0.1))
    pb.update_marks({"BTC/JPY": 200})
    pb.submit_order(_mkt("b2", Side.BUY, 0.1))

    pos = pb.get_positions()["BTC/JPY"]
    assert pos.qty == pytest.approx(0.2)
    assert pos.avg_price == pytest.approx(150.0)

    pb.submit_order(_mkt("s1", Side.SELL, 0.1))
    pb.submit_order(_mkt("s2", Side.SELL, 0.1))
    assert "BTC/JPY" not in pb.get_positions()  # flat
    assert pb.realized_pnl["BTC/JPY"] == pytest.approx(10.0)
    assert pb.equity()["JPY"] == pytest.approx(1_000_000_010.0)


def test_market_slippage_costs_money():
    pb = PaperBroker(INSTRUMENTS, {"JPY": 1_000_000_000}, venue=Venue.PAPER, slippage_bps=5)
    pb.update_marks({"BTC/JPY": 5_000_000})
    pb.submit_order(_mkt("b", Side.BUY, 0.1))
    pb.submit_order(_mkt("s", Side.SELL, 0.1))
    # round-trip loses 2x slippage on 0.1 @ 5_000_000
    assert pb.equity()["JPY"] == pytest.approx(1_000_000_000 - 500.0)


def test_idempotent_client_order_id():
    pb = PaperBroker(INSTRUMENTS, {"JPY": 1_000_000_000}, venue=Venue.PAPER, slippage_bps=0)
    pb.update_marks({"BTC/JPY": 100})
    pb.submit_order(_mkt("dup", Side.BUY, 0.1))
    again = pb.submit_order(_mkt("dup", Side.BUY, 0.1))
    assert again.status is OrderStatus.FILLED
    assert len(pb.fills) == 1
    assert pb.get_positions()["BTC/JPY"].qty == pytest.approx(0.1)


def test_maker_rebate_on_bitbank_fees():
    pb = PaperBroker(INSTRUMENTS, {"JPY": 1_000_000_000}, venue=Venue.BITBANK)
    pb.update_marks({"BTC/JPY": 5_000_000})
    pb.submit_order(_limit("m", Side.BUY, 0.1, 5_000_000))  # marketable maker buy
    # notional 500_000; maker -2bps => +100 JPY rebate
    assert pb.cash["JPY"] == pytest.approx(1_000_000_000 - 500_000 + 100.0)
    assert len(pb.fills) == 1
    assert pb.fills[0].fee == pytest.approx(-100.0)


def test_resting_limit_fills_when_mark_crosses():
    pb = PaperBroker(INSTRUMENTS, {"JPY": 1_000_000_000}, venue=Venue.PAPER, slippage_bps=0)
    pb.update_marks({"BTC/JPY": 100})
    order = pb.submit_order(_limit("rest", Side.BUY, 0.1, 90))
    assert order.status is OrderStatus.SUBMITTED
    assert len(pb.get_open_orders()) == 1
    assert not pb.get_positions()

    pb.update_marks({"BTC/JPY": 90})  # crosses the limit
    assert len(pb.get_open_orders()) == 0
    assert pb.get_positions()["BTC/JPY"].qty == pytest.approx(0.1)
