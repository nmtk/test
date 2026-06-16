from engine.core.clock import utc_now
from engine.core.types import (
    AssetClass,
    Fill,
    Instrument,
    OrderStatus,
    Position,
    Side,
    Venue,
)


def test_side_sign():
    assert Side.BUY.sign == 1
    assert Side.SELL.sign == -1


def test_round_size_rounds_toward_zero():
    inst = Instrument("BTC/JPY", AssetClass.CRYPTO, Venue.BITBANK, size_step=0.0001)
    assert inst.round_size(0.123456) == 0.1234
    assert inst.round_size(-0.123456) == -0.1234
    assert inst.round_size(0.00005) == 0.0  # below one step


def test_round_size_no_step_is_identity():
    inst = Instrument("AAPL", AssetClass.EQUITY, Venue.ALPACA)
    assert inst.round_size(3.14159) == 3.14159


def test_position_flat_and_value():
    pos = Position(
        "BTC/JPY", qty=0.5, avg_price=100.0, venue=Venue.PAPER, asset_class=AssetClass.CRYPTO
    )
    assert not pos.is_flat
    assert pos.market_value(200.0) == 100.0


def test_fill_notional():
    f = Fill(
        utc_now(),
        "AAPL",
        Side.BUY,
        qty=10,
        price=190.0,
        fee=0.0,
        fee_currency="USD",
        venue=Venue.ALPACA,
    )
    assert f.notional == 1900.0


def test_order_status_terminal():
    assert OrderStatus.FILLED.is_terminal
    assert not OrderStatus.SUBMITTED.is_terminal
