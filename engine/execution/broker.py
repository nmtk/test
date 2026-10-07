"""Abstract broker/market-data interfaces, fee models, and idempotent reconciliation.

Two small interfaces keep concerns separate:

* :class:`MarketData` — read bars and quotes.
* :class:`Broker` — read account state and submit/cancel orders.

Live adapters implement both; the paper broker implements :class:`Broker` only and gets
its prices pushed in. The free function :func:`reconcile_orders` is the heart of safe
execution: it turns *target* positions into the minimal set of orders given the *current*
positions, with deterministic client order IDs so a delayed or duplicated run never
double-fills.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from engine.core.types import (
    AssetClass,
    Balance,
    Instrument,
    Order,
    OrderType,
    Position,
    Quote,
    Side,
    TimeInForce,
    Venue,
)


class MarketData(ABC):
    @abstractmethod
    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Return canonical OHLCV bars (see :mod:`engine.data.schema`)."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Return the latest quote for ``symbol``."""


class Broker(ABC):
    venue: Venue

    @abstractmethod
    def get_balances(self) -> dict[str, Balance]:
        """Currency -> balance."""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Symbol -> net position (signed qty)."""

    @abstractmethod
    def get_open_orders(self, symbol: str | None = None) -> list[Order]: ...

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order. Implementations must be idempotent on ``client_order_id``."""

    @abstractmethod
    def cancel_order(self, broker_order_id: str, symbol: str | None = None) -> None: ...


@dataclass(frozen=True, slots=True)
class FeeModel:
    """Linear fee model in basis points of notional. ``maker_bps`` may be negative (rebate)."""

    maker_bps: float
    taker_bps: float

    def fee(self, notional: float, is_maker: bool) -> float:
        bps = self.maker_bps if is_maker else self.taker_bps
        return abs(notional) * bps / 10_000.0


# Venue fee defaults (see plan): bitbank maker -0.02% / taker 0.12%; Alpaca commission-free.
# OANDA's retail pricing carries no commission — the whole cost is the bid/ask spread, so
# it belongs in slippage, measured from tick data (see engine/data/dukascopy.py) rather
# than assumed here.
FEE_MODELS: dict[Venue, FeeModel] = {
    Venue.BITBANK: FeeModel(maker_bps=-2.0, taker_bps=12.0),
    Venue.ALPACA: FeeModel(maker_bps=0.0, taker_bps=0.0),
    Venue.OANDA: FeeModel(maker_bps=0.0, taker_bps=0.0),
    Venue.PAPER: FeeModel(maker_bps=0.0, taker_bps=0.0),
}


# Research/backtest cost assumptions per asset class, as (fee_bps, slippage_bps) charged on
# each unit of traded weight. These are *not* the same thing as FEE_MODELS above: that
# table is what a venue charges, this one is what a backtest should assume before a venue
# is even chosen.
#
# The split matters. Crypto pays a real taker fee on top of a wide book. Spot FX pays no
# commission at all — the entire cost is the bid/ask spread, and on USD/JPY that is roughly
# 0.2-0.5 pips on a ~150 price, i.e. ~0.15-0.35bps. Charging crypto's 15bps to an FX
# strategy overstates its costs by an order of magnitude and rejects edges that are really
# there; measure the true figure per instrument with
# :func:`engine.data.dukascopy.spread_stats_bps` and override when it is known.
RESEARCH_COSTS_BPS: dict[AssetClass, tuple[float, float]] = {
    AssetClass.CRYPTO: (10.0, 5.0),
    AssetClass.EQUITY: (0.0, 5.0),
    AssetClass.FX: (0.0, 1.0),
}


def research_costs(asset_class: AssetClass) -> tuple[float, float]:
    """``(fee_bps, slippage_bps)`` to assume for ``asset_class`` when backtesting."""
    return RESEARCH_COSTS_BPS.get(asset_class, RESEARCH_COSTS_BPS[AssetClass.CRYPTO])


# Costs for strategies that fill on the bid/ask path (engine/backtest/intrabar.py). Those
# already pay the *measured* spread on every fill, so the figures above — which stand in
# for the spread in mid-price backtests — would charge it twice. What remains is the gap
# between the data's book and the execution venue's: the path is Dukascopy's quote,
# measured on USD/JPY at a median 0.6 pip (p90 0.9, Aug-Nov 2023), while a retail venue
# is on the order of a pip all-in for OANDA's standard pricing, or a raw spread plus
# commission. 0.3bps per fill (~0.45 pip at 150, ~0.9 pip a round trip) puts the all-in
# round trip near 1.5 pips — wider than that venue, deliberately. A venue known to be
# tighter is passed explicitly (`--fee-bps/--slippage-bps`); it is never assumed.
INTRABAR_COSTS_BPS: dict[AssetClass, tuple[float, float]] = {
    AssetClass.FX: (0.0, 0.3),
}


def intrabar_research_costs(asset_class: AssetClass) -> tuple[float, float]:
    """``(fee_bps, slippage_bps)`` on top of the measured spread, for intrabar fills.

    An asset class with no intrabar figure falls back to its mid-price costs, which
    double-count the spread — wrong, but in the direction that rejects, not invents.
    """
    return INTRABAR_COSTS_BPS.get(asset_class, research_costs(asset_class))


def make_client_order_id(prefix: str, symbol: str, bar_ts_ms: int, target_qty: float) -> str:
    """Deterministic client order id for a (run, symbol, bar, target) tuple.

    Determinism is what makes retries safe: the same intended trade always produces the
    same id, so the venue rejects a duplicate instead of placing a second order.
    """
    raw = f"{prefix}|{symbol}|{bar_ts_ms}|{target_qty:.10g}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def reconcile_orders(
    current: dict[str, Position],
    targets: dict[str, float],
    instruments: dict[str, Instrument],
    bar_ts_ms: int,
    *,
    prefix: str = "exec",
    limit_prices: dict[str, float] | None = None,
    min_trade_notional: float = 0.0,
    ref_prices: dict[str, float] | None = None,
) -> list[Order]:
    """Compute the minimal orders to move from ``current`` to ``targets`` positions.

    ``targets`` maps symbol -> desired signed quantity. Symbols present in ``current`` but
    absent from ``targets`` are treated as a target of zero (flatten). Orders smaller than
    the instrument's size step (after rounding) or below ``min_trade_notional`` are skipped.
    """
    limit_prices = limit_prices or {}
    ref_prices = ref_prices or {}
    symbols = set(current) | set(targets)
    orders: list[Order] = []

    for symbol in sorted(symbols):
        instrument = instruments.get(symbol)
        if instrument is None:
            continue
        current_qty = current[symbol].qty if symbol in current else 0.0
        target_qty = targets.get(symbol, 0.0)
        delta = target_qty - current_qty
        delta = instrument.round_size(delta)
        if delta == 0.0:
            continue

        ref_price = ref_prices.get(symbol) or limit_prices.get(symbol) or 0.0
        if min_trade_notional > 0.0 and ref_price > 0.0:
            if abs(delta) * ref_price < min_trade_notional:
                continue

        side = Side.BUY if delta > 0 else Side.SELL
        limit_price = limit_prices.get(symbol)
        order = Order(
            client_order_id=make_client_order_id(prefix, symbol, bar_ts_ms, target_qty),
            symbol=symbol,
            side=side,
            qty=abs(delta),
            type=OrderType.LIMIT if limit_price is not None else OrderType.MARKET,
            limit_price=limit_price,
            tif=TimeInForce.POST_ONLY if limit_price is not None else TimeInForce.GTC,
            reduce_only=(target_qty == 0.0),
            venue=instrument.venue,
        )
        orders.append(order)
    return orders
