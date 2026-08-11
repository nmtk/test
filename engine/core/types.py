"""The stable trading vocabulary shared by every layer of the engine.

These types are deliberately small, immutable where practical, and broker-agnostic.
Brokers (bitbank, Alpaca, paper) translate to/from their native representations at the
edges; everything inside the engine speaks only these types.

Conventions
-----------
* All timestamps are timezone-aware ``datetime`` in **UTC**.
* A :class:`Bar` timestamp is the bar's **open** time (ccxt convention). Only *closed*
  bars are ever stored or traded on.
* :class:`Position` quantity is **signed**: positive = long, negative = short.
* Prices/quantities are ``float``. Order sizing must round to a venue's size/price step
  before submission (see :class:`Instrument`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class AssetClass(StrEnum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    FX = "fx"


class Venue(StrEnum):
    BITBANK = "bitbank"
    ALPACA = "alpaca"
    OANDA = "oanda"
    BINANCE = "binance"  # research data source only (not an execution venue here)
    DUKASCOPY = "dukascopy"  # research data source only (free historical FX ticks)
    PAPER = "paper"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    GTC = "gtc"
    DAY = "day"
    IOC = "ioc"
    POST_ONLY = "post_only"  # maker-only; reject if it would take liquidity


class OrderStatus(StrEnum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradable instrument, identified by a canonical symbol.

    Canonical symbols: ``"BTC/JPY"`` for crypto (ccxt unified style), ``"AAPL"`` for
    equities. ``venue`` is the execution venue; research-only data may use a different
    source symbol handled by the data layer.
    """

    symbol: str
    asset_class: AssetClass
    venue: Venue
    base: str | None = None
    quote: str | None = None
    min_size: float = 0.0
    size_step: float = 0.0
    price_step: float = 0.0
    min_notional: float = 0.0

    def round_size(self, size: float) -> float:
        """Round an order size *down* to the venue's size step (toward zero).

        Uses a small epsilon and a final round to avoid binary floating-point dust (e.g.
        ``0.1234/0.0001`` landing on ``1233.9999...``), which real venues reject.
        """
        if self.size_step and self.size_step > 0:
            steps = int(abs(size) / self.size_step + 1e-9)
            rounded = round(steps * self.size_step, 10)
            return rounded if size >= 0 else -rounded
        return size


@dataclass(frozen=True, slots=True)
class Bar:
    """A single closed OHLCV bar. ``ts`` is the bar's open time in UTC."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Quote:
    ts: datetime
    bid: float
    ask: float
    last: float

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last


@dataclass(slots=True)
class Order:
    """An order request/record in engine vocabulary.

    ``client_order_id`` makes submission idempotent: re-submitting the same id is a
    no-op at the venue, so delayed or duplicated executor runs never double-fill.
    """

    client_order_id: str
    symbol: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    tif: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    venue: Venue | None = None
    # Populated once the order is known to a broker:
    broker_order_id: str | None = None
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0


@dataclass(frozen=True, slots=True)
class Fill:
    """A single execution (partial or full)."""

    ts: datetime
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float
    fee_currency: str
    venue: Venue
    client_order_id: str | None = None
    broker_order_id: str | None = None

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass(frozen=True, slots=True)
class Position:
    """A net position. ``qty`` is signed (positive long, negative short)."""

    symbol: str
    qty: float
    avg_price: float
    venue: Venue
    asset_class: AssetClass

    @property
    def is_flat(self) -> bool:
        return self.qty == 0.0

    def market_value(self, price: float) -> float:
        return self.qty * price


@dataclass(frozen=True, slots=True)
class Balance:
    currency: str
    total: float
    free: float
    used: float = 0.0


@dataclass(slots=True)
class AccountState:
    """A point-in-time snapshot of an account at a single venue."""

    venue: Venue
    balances: dict[str, Balance] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)

    def position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)
