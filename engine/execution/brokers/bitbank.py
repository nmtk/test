"""bitbank live adapter (spot, JPY) via ccxt.

Trade-only: this class never calls deposit/withdrawal endpoints. On a spot venue a
"position" in ``BASE/QUOTE`` is simply the held BASE balance, so positions are
synthesized from balances using the instrument's base currency.
"""

from __future__ import annotations

import pandas as pd

from engine.core.clock import utc_now
from engine.core.types import (
    Balance,
    Instrument,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
    TimeInForce,
    Venue,
)
from engine.data.schema import OHLCV_COLUMNS, empty_ohlcv, normalize_ohlcv
from engine.execution.broker import Broker, MarketData


class BitbankBroker(Broker, MarketData):
    venue = Venue.BITBANK

    def __init__(self, api_key: str, api_secret: str, instruments: dict[str, Instrument]):
        self.instruments = dict(instruments)
        self._api_key = api_key
        self._api_secret = api_secret
        self._client = None

    def _c(self):
        if self._client is None:
            import ccxt

            self._client = ccxt.bitbank(
                {"apiKey": self._api_key, "secret": self._api_secret, "enableRateLimit": True}
            )
        return self._client

    def get_balances(self) -> dict[str, Balance]:
        raw = self._c().fetch_balance()
        out: dict[str, Balance] = {}
        for currency, total in (raw.get("total") or {}).items():
            free = (raw.get("free") or {}).get(currency, 0.0) or 0.0
            used = (raw.get("used") or {}).get(currency, 0.0) or 0.0
            out[currency] = Balance(currency, total=total or 0.0, free=free, used=used)
        return out

    def get_positions(self) -> dict[str, Position]:
        balances = self.get_balances()
        out: dict[str, Position] = {}
        for symbol, instrument in self.instruments.items():
            base = instrument.base or symbol.split("/")[0]
            bal = balances.get(base)
            if bal and bal.total != 0.0:
                out[symbol] = Position(
                    symbol=symbol,
                    qty=bal.total,
                    avg_price=0.0,  # spot venue gives no cost basis; tracked in our ledger
                    venue=self.venue,
                    asset_class=instrument.asset_class,
                )
        return out

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        raw = self._c().fetch_open_orders(symbol)
        return [self._to_order(o) for o in raw]

    def submit_order(self, order: Order) -> Order:
        params: dict = {}
        if order.tif is TimeInForce.POST_ONLY:
            params["post_only"] = True
        price = order.limit_price if order.type is OrderType.LIMIT else None
        result = self._c().create_order(
            order.symbol,
            order.type.value,
            order.side.value,
            order.qty,
            price,
            params,
        )
        order.broker_order_id = str(result.get("id"))
        order.status = OrderStatus.SUBMITTED
        return order

    def cancel_order(self, broker_order_id: str, symbol: str | None = None) -> None:
        self._c().cancel_order(broker_order_id, symbol)

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        rows = self._c().fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not rows:
            return empty_ohlcv()
        return normalize_ohlcv(pd.DataFrame(rows, columns=OHLCV_COLUMNS))

    def get_quote(self, symbol: str) -> Quote:
        t = self._c().fetch_ticker(symbol)
        return Quote(
            ts=utc_now(),
            bid=t.get("bid") or 0.0,
            ask=t.get("ask") or 0.0,
            last=t.get("last") or 0.0,
        )

    def _to_order(self, raw: dict) -> Order:
        return Order(
            client_order_id=str(raw.get("clientOrderId") or raw.get("id")),
            symbol=raw.get("symbol"),
            side=Side(raw.get("side")),
            qty=float(raw.get("amount") or 0.0),
            type=OrderType(raw.get("type") or "limit"),
            limit_price=raw.get("price"),
            venue=self.venue,
            broker_order_id=str(raw.get("id")),
            filled_qty=float(raw.get("filled") or 0.0),
        )
