"""OANDA v20 live adapter (spot FX) — the execution venue for USD/JPY strategies.

Pure REST, like every other adapter here, so no always-on process is required: the
executor wakes on its cron, reconciles, and exits. Defaults to the **practice**
environment; live requires both ``OANDA_PRACTICE=false`` and the engine's own
``TRADE_LIVE_ENABLED``, so a single misconfigured variable cannot reach real money.

Credentials must be a **trade-only** token. OANDA tokens cannot initiate withdrawals, but
they can transfer between an account's own sub-accounts, so use a token scoped to the one
account you intend to trade.

Two mappings deserve attention:

* **Symbols.** The engine says ``USD/JPY``; v20 says ``USD_JPY``.
* **Sizes.** v20 has no "position" object to set — you submit signed *units* of the base
  currency, and an integer at that. The engine's reconciler already computes the signed
  delta between the current and target position, which is exactly what v20 wants, so the
  translation is direct. Fractional units are rejected by the venue, so sizes are rounded
  toward zero (never up, which would breach a risk cap).
"""

from __future__ import annotations

from datetime import UTC, datetime

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
from engine.data.schema import empty_ohlcv, normalize_ohlcv
from engine.execution.broker import Broker, MarketData

PRACTICE_HOST = "https://api-fxpractice.oanda.com"
LIVE_HOST = "https://api-fxtrade.oanda.com"

# Engine timeframe -> v20 granularity. v20 has no 4h, so it is derived by resampling.
_GRANULARITY = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "1d": "D",
}
_MAX_CANDLES = 5000  # v20's per-request ceiling


def to_oanda_symbol(symbol: str) -> str:
    """``"USD/JPY"`` -> ``"USD_JPY"``."""
    return symbol.replace("/", "_").upper()


def from_oanda_symbol(instrument: str) -> str:
    """``"USD_JPY"`` -> ``"USD/JPY"``."""
    return instrument.replace("_", "/").upper()


class OandaBroker(Broker, MarketData):
    """Broker + market data for spot FX via the v20 REST API."""

    venue = Venue.OANDA

    def __init__(
        self,
        api_token: str | None,
        account_id: str | None,
        practice: bool = True,
        timeout: int = 30,
    ):
        self._token = api_token
        self.account_id = account_id
        self.practice = practice
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings) -> OandaBroker:
        oanda = getattr(settings, "oanda", None)
        return cls(
            api_token=getattr(oanda, "api_key", None),
            account_id=getattr(oanda, "api_secret", None),
            practice=getattr(settings, "oanda_practice", True),
        )

    @property
    def host(self) -> str:
        return PRACTICE_HOST if self.practice else LIVE_HOST

    @property
    def is_configured(self) -> bool:
        return bool(self._token and self.account_id)

    # -- transport ---------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs):
        import requests

        if not self.is_configured:
            raise RuntimeError(
                "OANDA credentials missing: set OANDA_API_TOKEN and OANDA_ACCOUNT_ID"
            )
        response = requests.request(
            method,
            f"{self.host}{path}",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                # v20 returns RFC3339 timestamps by default; UNIX is unambiguous.
                "Accept-Datetime-Format": "UNIX",
            },
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json() if response.content else {}

    def _account_path(self, suffix: str = "") -> str:
        return f"/v3/accounts/{self.account_id}{suffix}"

    # -- account state -----------------------------------------------------------------

    def get_balances(self) -> dict[str, Balance]:
        summary = self._request("GET", self._account_path("/summary")).get("account", {})
        currency = summary.get("currency", "USD")
        nav = float(summary.get("NAV", summary.get("balance", 0.0)))
        used = float(summary.get("marginUsed", 0.0))
        available = float(summary.get("marginAvailable", nav - used))
        return {currency: Balance(currency, total=nav, free=available, used=used)}

    def get_positions(self) -> dict[str, Position]:
        """Net signed position per instrument.

        v20 reports long and short legs separately (they can both be open on a hedging
        account), so they are netted into the engine's single signed quantity.
        """
        payload = self._request("GET", self._account_path("/openPositions"))
        out: dict[str, Position] = {}
        for raw in payload.get("positions", []):
            long_units = float(raw.get("long", {}).get("units", 0.0))
            short_units = float(raw.get("short", {}).get("units", 0.0))  # already negative
            qty = long_units + short_units
            if qty == 0.0:
                continue
            leg = raw.get("long" if qty > 0 else "short", {})
            out[from_oanda_symbol(raw["instrument"])] = Position(
                symbol=from_oanda_symbol(raw["instrument"]),
                qty=qty,
                avg_price=float(leg.get("averagePrice", 0.0) or 0.0),
                venue=self.venue,
                asset_class=AssetClass.FX,
            )
        return out

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        payload = self._request("GET", self._account_path("/pendingOrders"))
        orders = [self._to_order(raw) for raw in payload.get("orders", [])]
        orders = [o for o in orders if o.symbol]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    # -- order flow --------------------------------------------------------------------

    def submit_order(self, order: Order) -> Order:
        """Submit a signed-units order, idempotent on ``client_order_id``.

        v20 enforces uniqueness of ``clientExtensions.id`` per account, so a retried or
        duplicated executor run is rejected by the venue rather than double-filling.
        """
        units = int(order.qty) * order.side.sign
        if units == 0:
            # Rounded away to nothing — submitting 0 units is an error at the venue.
            order.status = OrderStatus.REJECTED
            return order

        body: dict = {
            "order": {
                "instrument": to_oanda_symbol(order.symbol),
                "units": str(units),
                "type": "LIMIT" if order.type is OrderType.LIMIT else "MARKET",
                "clientExtensions": {"id": order.client_order_id},
            }
        }
        if order.type is OrderType.LIMIT:
            body["order"]["price"] = f"{order.limit_price}"
            # POST_ONLY in engine vocabulary == "never take liquidity" == v20's maker-only
            # limit. Anything else is a plain GTC limit.
            body["order"]["timeInForce"] = "GTC"
            if order.tif is TimeInForce.POST_ONLY:
                body["order"]["positionFill"] = "DEFAULT"
        else:
            body["order"]["timeInForce"] = "FOK"
        if order.reduce_only:
            body["order"]["positionFill"] = "REDUCE_ONLY"

        payload = self._request("POST", self._account_path("/orders"), json=body)
        created = payload.get("orderCreateTransaction", {}) or {}
        filled = payload.get("orderFillTransaction", {}) or {}
        order.broker_order_id = str(created.get("id") or filled.get("orderID") or "")
        if filled:
            order.status = OrderStatus.FILLED
            order.filled_qty = abs(float(filled.get("units", 0.0)))
            order.avg_fill_price = float(filled.get("price", 0.0) or 0.0)
        elif payload.get("orderCancelTransaction"):
            order.status = OrderStatus.REJECTED
        else:
            order.status = OrderStatus.SUBMITTED
        return order

    def cancel_order(self, broker_order_id: str, symbol: str | None = None) -> None:
        self._request("PUT", self._account_path(f"/orders/{broker_order_id}/cancel"))

    # -- market data -------------------------------------------------------------------

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Mid-price candles. Only **complete** candles are returned.

        The forming candle is dropped: trading a bar that has not closed is exactly the
        look-ahead the rest of the engine is built to prevent.
        """
        granularity = _GRANULARITY.get(timeframe)
        if granularity is None:
            raise ValueError(
                f"unsupported OANDA timeframe {timeframe!r}; expected one of {sorted(_GRANULARITY)}"
            )
        params: dict = {
            "granularity": granularity,
            "price": "M",  # midpoint
            "count": min(limit or _MAX_CANDLES, _MAX_CANDLES),
        }
        if since_ms is not None:
            params["from"] = f"{since_ms / 1000:.6f}"
            params.pop("count", None)
        payload = self._request(
            "GET", f"/v3/instruments/{to_oanda_symbol(symbol)}/candles", params=params
        )
        rows = []
        for candle in payload.get("candles", []):
            if not candle.get("complete", False):
                continue
            mid = candle.get("mid", {})
            rows.append(
                [
                    int(float(candle["time"]) * 1000),
                    float(mid["o"]),
                    float(mid["h"]),
                    float(mid["l"]),
                    float(mid["c"]),
                    float(candle.get("volume", 0.0)),
                ]
            )
        if not rows:
            return empty_ohlcv()
        return normalize_ohlcv(
            pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        )

    def get_quote(self, symbol: str) -> Quote:
        payload = self._request(
            "GET",
            self._account_path("/pricing"),
            params={"instruments": to_oanda_symbol(symbol)},
        )
        prices = payload.get("prices", [])
        if not prices:
            return Quote(ts=utc_now(), bid=0.0, ask=0.0, last=0.0)
        price = prices[0]
        bid = float((price.get("bids") or [{}])[0].get("price", 0.0) or 0.0)
        ask = float((price.get("asks") or [{}])[0].get("price", 0.0) or 0.0)
        ts = price.get("time")
        when = datetime.fromtimestamp(float(ts), tz=UTC) if ts else utc_now()
        return Quote(ts=when, bid=bid, ask=ask, last=(bid + ask) / 2.0 if bid and ask else 0.0)

    # -- translation -------------------------------------------------------------------

    def _to_order(self, raw: dict) -> Order:
        instrument = raw.get("instrument")
        units = float(raw.get("units", 0.0) or 0.0)
        price = raw.get("price")
        return Order(
            client_order_id=str((raw.get("clientExtensions") or {}).get("id") or raw.get("id", "")),
            symbol=from_oanda_symbol(instrument) if instrument else "",
            side=Side.BUY if units >= 0 else Side.SELL,
            qty=abs(units),
            type=OrderType.LIMIT if price else OrderType.MARKET,
            limit_price=float(price) if price else None,
            venue=self.venue,
            broker_order_id=str(raw.get("id", "")),
            filled_qty=abs(float(raw.get("filledUnits", 0.0) or 0.0)),
        )
