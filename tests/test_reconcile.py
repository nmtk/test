from engine.core.types import AssetClass, Instrument, OrderType, Position, Side, TimeInForce, Venue
from engine.execution.broker import make_client_order_id, reconcile_orders

BTC = Instrument("BTC/JPY", AssetClass.CRYPTO, Venue.BITBANK, size_step=0.0001)
INSTRUMENTS = {"BTC/JPY": BTC}


def _pos(qty):
    return Position(
        "BTC/JPY", qty=qty, avg_price=100.0, venue=Venue.BITBANK, asset_class=AssetClass.CRYPTO
    )


def test_open_from_flat():
    orders = reconcile_orders({}, {"BTC/JPY": 0.5}, INSTRUMENTS, bar_ts_ms=1000)
    assert len(orders) == 1
    o = orders[0]
    assert o.side is Side.BUY
    assert o.qty == 0.5
    assert o.type is OrderType.MARKET
    assert not o.reduce_only


def test_flatten_when_target_missing():
    orders = reconcile_orders({"BTC/JPY": _pos(0.5)}, {}, INSTRUMENTS, bar_ts_ms=1000)
    assert len(orders) == 1
    assert orders[0].side is Side.SELL
    assert orders[0].qty == 0.5
    assert orders[0].reduce_only


def test_no_order_when_already_on_target():
    orders = reconcile_orders({"BTC/JPY": _pos(0.5)}, {"BTC/JPY": 0.5}, INSTRUMENTS, bar_ts_ms=1000)
    assert orders == []


def test_deterministic_client_order_id():
    a = reconcile_orders({}, {"BTC/JPY": 0.5}, INSTRUMENTS, bar_ts_ms=1000)
    b = reconcile_orders({}, {"BTC/JPY": 0.5}, INSTRUMENTS, bar_ts_ms=1000)
    assert a[0].client_order_id == b[0].client_order_id
    # different bar => different id
    c = reconcile_orders({}, {"BTC/JPY": 0.5}, INSTRUMENTS, bar_ts_ms=2000)
    assert a[0].client_order_id != c[0].client_order_id


def test_min_trade_notional_skips_dust():
    orders = reconcile_orders(
        {},
        {"BTC/JPY": 0.0001},
        INSTRUMENTS,
        bar_ts_ms=1000,
        ref_prices={"BTC/JPY": 100.0},
        min_trade_notional=1.0,  # 0.0001 * 100 = 0.01 < 1.0
    )
    assert orders == []


def test_limit_price_makes_post_only_order():
    orders = reconcile_orders(
        {}, {"BTC/JPY": 0.5}, INSTRUMENTS, bar_ts_ms=1000, limit_prices={"BTC/JPY": 4_900_000}
    )
    assert orders[0].type is OrderType.LIMIT
    assert orders[0].tif is TimeInForce.POST_ONLY
    assert orders[0].limit_price == 4_900_000


def test_make_client_order_id_stable():
    assert make_client_order_id("exec", "BTC/JPY", 1000, 0.5) == make_client_order_id(
        "exec", "BTC/JPY", 1000, 0.5
    )
