"""OANDA v20 adapter tests — translation, netting, and safety defaults.

The v20 API is stubbed at the transport boundary (``_request``) with real response shapes,
so the tests cover exactly the code the engine owns: symbol/unit translation, position
netting, and the mapping between engine and venue vocabulary.
"""

from __future__ import annotations

import pytest

from engine.core.types import (
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    Venue,
)
from engine.execution.brokers.oanda import (
    LIVE_HOST,
    PRACTICE_HOST,
    OandaBroker,
    from_oanda_symbol,
    to_oanda_symbol,
)


class _StubBroker(OandaBroker):
    """An OandaBroker whose transport returns canned v20 payloads."""

    def __init__(self, responses: dict | None = None, **kwargs):
        super().__init__(api_token="token", account_id="001-001-1234567-001", **kwargs)
        self.responses = responses or {}
        self.calls: list[tuple[str, str, dict]] = []

    def _request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return self.responses.get(path, self.responses.get(f"{method} {path}", {}))


# -- translation -------------------------------------------------------------------------


def test_symbol_translation_round_trips():
    assert to_oanda_symbol("USD/JPY") == "USD_JPY"
    assert from_oanda_symbol("USD_JPY") == "USD/JPY"
    assert from_oanda_symbol(to_oanda_symbol("EUR/USD")) == "EUR/USD"


def test_venue_and_practice_defaults():
    broker = OandaBroker(api_token="t", account_id="a")
    assert broker.venue is Venue.OANDA
    assert broker.practice is True, "practice must be the default, never live"
    assert broker.host == PRACTICE_HOST
    assert OandaBroker(api_token="t", account_id="a", practice=False).host == LIVE_HOST


def test_missing_credentials_raise_rather_than_silently_no_op():
    broker = OandaBroker(api_token=None, account_id=None)
    assert not broker.is_configured
    with pytest.raises(RuntimeError, match="OANDA credentials missing"):
        broker.get_balances()


# -- account state -----------------------------------------------------------------------


def test_balances_report_nav_and_available_margin():
    broker = _StubBroker(
        {
            "/v3/accounts/001-001-1234567-001/summary": {
                "account": {
                    "currency": "JPY",
                    "NAV": "1050000.0",
                    "balance": "1000000.0",
                    "marginUsed": "50000.0",
                    "marginAvailable": "1000000.0",
                }
            }
        }
    )
    balances = broker.get_balances()
    assert balances["JPY"].total == pytest.approx(1_050_000.0)
    assert balances["JPY"].used == pytest.approx(50_000.0)
    assert balances["JPY"].free == pytest.approx(1_000_000.0)


def test_long_and_short_legs_are_netted_into_one_signed_position():
    """v20 can carry both legs at once; the engine speaks a single signed quantity."""
    broker = _StubBroker(
        {
            "/v3/accounts/001-001-1234567-001/openPositions": {
                "positions": [
                    {
                        "instrument": "USD_JPY",
                        "long": {"units": "10000", "averagePrice": "151.20"},
                        "short": {"units": "-4000", "averagePrice": "151.50"},
                    }
                ]
            }
        }
    )
    positions = broker.get_positions()
    assert set(positions) == {"USD/JPY"}
    assert positions["USD/JPY"].qty == pytest.approx(6000.0)
    assert positions["USD/JPY"].avg_price == pytest.approx(151.20)


def test_net_flat_position_is_omitted():
    broker = _StubBroker(
        {
            "/v3/accounts/001-001-1234567-001/openPositions": {
                "positions": [
                    {
                        "instrument": "USD_JPY",
                        "long": {"units": "5000", "averagePrice": "151.20"},
                        "short": {"units": "-5000", "averagePrice": "151.50"},
                    }
                ]
            }
        }
    )
    assert broker.get_positions() == {}


def test_short_only_position_is_negative():
    broker = _StubBroker(
        {
            "/v3/accounts/001-001-1234567-001/openPositions": {
                "positions": [
                    {
                        "instrument": "USD_JPY",
                        "long": {"units": "0"},
                        "short": {"units": "-2500", "averagePrice": "151.50"},
                    }
                ]
            }
        }
    )
    assert broker.get_positions()["USD/JPY"].qty == pytest.approx(-2500.0)


# -- order flow --------------------------------------------------------------------------


def _order(side: Side, qty: float, **kwargs) -> Order:
    return Order(
        client_order_id="exec-abc123",
        symbol="USD/JPY",
        side=side,
        qty=qty,
        venue=Venue.OANDA,
        **kwargs,
    )


def test_sell_order_is_submitted_as_negative_units():
    broker = _StubBroker(
        {
            "/v3/accounts/001-001-1234567-001/orders": {
                "orderCreateTransaction": {"id": "77"},
                "orderFillTransaction": {"units": "-5000", "price": "151.234", "orderID": "77"},
            }
        }
    )
    result = broker.submit_order(_order(Side.SELL, 5000))
    body = broker.calls[0][2]["json"]["order"]

    assert body["units"] == "-5000", "a sell must be negative units"
    assert body["instrument"] == "USD_JPY"
    assert result.status is OrderStatus.FILLED
    assert result.filled_qty == pytest.approx(5000.0)
    assert result.avg_fill_price == pytest.approx(151.234)


def test_buy_order_is_positive_units_and_carries_the_client_id():
    broker = _StubBroker(
        {"/v3/accounts/001-001-1234567-001/orders": {"orderCreateTransaction": {"id": "88"}}}
    )
    result = broker.submit_order(_order(Side.BUY, 1000))
    body = broker.calls[0][2]["json"]["order"]

    assert body["units"] == "1000"
    # Idempotency: v20 rejects a duplicate clientExtensions.id, so a retried run cannot
    # double-fill.
    assert body["clientExtensions"]["id"] == "exec-abc123"
    assert result.status is OrderStatus.SUBMITTED
    assert result.broker_order_id == "88"


def test_fractional_units_round_toward_zero():
    """v20 rejects fractional units; rounding up would breach a risk cap."""
    broker = _StubBroker(
        {"/v3/accounts/001-001-1234567-001/orders": {"orderCreateTransaction": {"id": "1"}}}
    )
    broker.submit_order(_order(Side.BUY, 1000.9))
    assert broker.calls[0][2]["json"]["order"]["units"] == "1000"


def test_order_rounding_to_nothing_is_rejected_not_sent():
    broker = _StubBroker()
    result = broker.submit_order(_order(Side.BUY, 0.4))
    assert result.status is OrderStatus.REJECTED
    assert broker.calls == [], "a zero-unit order must never reach the venue"


def test_reduce_only_is_marked_for_the_venue():
    broker = _StubBroker(
        {"/v3/accounts/001-001-1234567-001/orders": {"orderCreateTransaction": {"id": "2"}}}
    )
    broker.submit_order(_order(Side.SELL, 1000, reduce_only=True))
    assert broker.calls[0][2]["json"]["order"]["positionFill"] == "REDUCE_ONLY"


def test_limit_order_sends_a_price():
    broker = _StubBroker(
        {"/v3/accounts/001-001-1234567-001/orders": {"orderCreateTransaction": {"id": "3"}}}
    )
    broker.submit_order(
        _order(Side.BUY, 1000, type=OrderType.LIMIT, limit_price=150.5, tif=TimeInForce.POST_ONLY)
    )
    body = broker.calls[0][2]["json"]["order"]
    assert body["type"] == "LIMIT"
    assert float(body["price"]) == pytest.approx(150.5)


def test_cancelled_order_is_reported_as_rejected():
    broker = _StubBroker(
        {
            "/v3/accounts/001-001-1234567-001/orders": {
                "orderCreateTransaction": {"id": "4"},
                "orderCancelTransaction": {"reason": "INSUFFICIENT_MARGIN"},
            }
        }
    )
    assert broker.submit_order(_order(Side.BUY, 1000)).status is OrderStatus.REJECTED


def test_open_orders_are_translated():
    broker = _StubBroker(
        {
            "/v3/accounts/001-001-1234567-001/pendingOrders": {
                "orders": [
                    {
                        "id": "42",
                        "instrument": "USD_JPY",
                        "units": "-3000",
                        "price": "152.000",
                        "clientExtensions": {"id": "exec-deadbeef"},
                    }
                ]
            }
        }
    )
    orders = broker.get_open_orders()
    assert len(orders) == 1
    assert orders[0].side is Side.SELL
    assert orders[0].qty == pytest.approx(3000.0)
    assert orders[0].type is OrderType.LIMIT
    assert orders[0].client_order_id == "exec-deadbeef"
    assert broker.get_open_orders(symbol="EUR/USD") == []


# -- market data -------------------------------------------------------------------------


def test_incomplete_candle_is_dropped():
    """The forming bar must never reach a strategy — that is look-ahead by another name."""
    broker = _StubBroker(
        {
            "/v3/instruments/USD_JPY/candles": {
                "candles": [
                    {
                        "time": "1709640000.000000000",
                        "complete": True,
                        "volume": 120,
                        "mid": {"o": "151.0", "h": "151.5", "l": "150.8", "c": "151.2"},
                    },
                    {
                        "time": "1709640300.000000000",
                        "complete": False,
                        "volume": 4,
                        "mid": {"o": "151.2", "h": "151.3", "l": "151.1", "c": "151.25"},
                    },
                ]
            }
        }
    )
    bars = broker.fetch_ohlcv("USD/JPY", "5m")
    assert len(bars) == 1
    assert bars["ts"].iloc[0] == 1_709_640_000_000
    assert bars["close"].iloc[0] == pytest.approx(151.2)


def test_unsupported_timeframe_is_rejected_loudly():
    with pytest.raises(ValueError, match="unsupported OANDA timeframe"):
        _StubBroker().fetch_ohlcv("USD/JPY", "4h")


def test_no_candles_returns_canonical_empty_frame():
    bars = _StubBroker({"/v3/instruments/USD_JPY/candles": {"candles": []}}).fetch_ohlcv(
        "USD/JPY", "5m"
    )
    assert len(bars) == 0
    assert list(bars.columns) == ["ts", "open", "high", "low", "close", "volume"]


def test_quote_reads_top_of_book():
    broker = _StubBroker(
        {
            "/v3/accounts/001-001-1234567-001/pricing": {
                "prices": [
                    {
                        "time": "1709640000.000000000",
                        "bids": [{"price": "151.230"}],
                        "asks": [{"price": "151.234"}],
                    }
                ]
            }
        }
    )
    quote = broker.get_quote("USD/JPY")
    assert quote.bid == pytest.approx(151.230)
    assert quote.ask == pytest.approx(151.234)
    assert quote.mid == pytest.approx(151.232)


def test_quote_with_no_prices_is_zero_not_an_exception():
    quote = _StubBroker({"/v3/accounts/001-001-1234567-001/pricing": {"prices": []}}).get_quote(
        "USD/JPY"
    )
    assert quote.bid == 0.0 and quote.ask == 0.0


# -- settings wiring ---------------------------------------------------------------------


def test_from_settings_reads_the_configured_credentials():
    from engine.core.config import load_settings

    broker = OandaBroker.from_settings(load_settings())
    assert broker.venue is Venue.OANDA
    # No credentials in the test environment, so it must report itself unconfigured
    # rather than pretending to be ready.
    assert broker.is_configured is False
    assert broker.practice is True
