"""Alpaca live adapter (US equities/ETFs) via alpaca-py.

Trade-only: no transfer/journal endpoints are called. Defaults to the paper endpoint and
the free IEX data feed; pass ``paper=False`` for live. Alpaca has no POST-ONLY for
equities, so POST_ONLY maps to a plain limit order.
"""

from __future__ import annotations

import pandas as pd

from engine.core.clock import utc_now
from engine.core.types import (
    AssetClass,
    Balance,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
    TimeInForce,
    Venue,
)
from engine.data.sources import AlpacaSource
from engine.execution.broker import Broker, MarketData


class AlpacaBroker(Broker, MarketData):
    venue = Venue.ALPACA

    def __init__(self, api_key: str, api_secret: str, paper: bool = True, feed: str = "iex"):
        self._api_key = api_key
        self._api_secret = api_secret
        self.paper = paper
        self._trading = None
        self._data = AlpacaSource(api_key, api_secret, feed=feed)
        self._quotes = None

    def _t(self):
        if self._trading is None:
            from alpaca.trading.client import TradingClient

            self._trading = TradingClient(self._api_key, self._api_secret, paper=self.paper)
        return self._trading

    def get_balances(self) -> dict[str, Balance]:
        acct = self._t().get_account()
        cash = float(acct.cash)
        equity = float(acct.equity)
        return {"USD": Balance("USD", total=equity, free=cash, used=equity - cash)}

    def get_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self._t().get_all_positions():
            qty = float(p.qty)  # alpaca reports signed qty for short positions
            out[p.symbol] = Position(
                symbol=p.symbol,
                qty=qty,
                avg_price=float(p.avg_entry_price),
                venue=self.venue,
                asset_class=AssetClass.EQUITY,
            )
        return out

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        symbols = [symbol] if symbol else None
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=symbols)
        return [self._to_order(o) for o in self._t().get_orders(req)]

    def submit_order(self, order: Order) -> Order:
        from alpaca.trading.enums import OrderSide
        from alpaca.trading.enums import TimeInForce as AlpacaTIF
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL
        tif = AlpacaTIF.DAY if order.tif is TimeInForce.DAY else AlpacaTIF.GTC
        common = dict(
            symbol=order.symbol,
            qty=order.qty,
            side=side,
            time_in_force=tif,
            client_order_id=order.client_order_id,
        )
        if order.type is OrderType.LIMIT:
            req = LimitOrderRequest(limit_price=order.limit_price, **common)
        else:
            req = MarketOrderRequest(**common)
        result = self._t().submit_order(req)
        order.broker_order_id = str(result.id)
        order.status = OrderStatus.SUBMITTED
        return order

    def cancel_order(self, broker_order_id: str, symbol: str | None = None) -> None:
        self._t().cancel_order_by_id(broker_order_id)

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        return self._data.fetch_ohlcv(symbol, timeframe, since_ms=since_ms, limit=limit)

    def get_quote(self, symbol: str) -> Quote:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        if self._quotes is None:
            self._quotes = StockHistoricalDataClient(self._api_key, self._api_secret)
        q = self._quotes.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[
            symbol
        ]
        bid = float(q.bid_price or 0.0)
        ask = float(q.ask_price or 0.0)
        return Quote(ts=utc_now(), bid=bid, ask=ask, last=(bid + ask) / 2 if bid and ask else 0.0)

    def _to_order(self, raw) -> Order:
        return Order(
            client_order_id=str(raw.client_order_id),
            symbol=raw.symbol,
            side=Side.BUY if str(raw.side).endswith("buy") else Side.SELL,
            qty=float(raw.qty or 0.0),
            type=OrderType.LIMIT if raw.limit_price else OrderType.MARKET,
            limit_price=float(raw.limit_price) if raw.limit_price else None,
            venue=self.venue,
            broker_order_id=str(raw.id),
            filled_qty=float(raw.filled_qty or 0.0),
        )
