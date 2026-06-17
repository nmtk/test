"""Paper broker: a deterministic fill simulator used for incubation and tests.

It runs the *same* execution code path as live trading; only the broker differs. Prices
are pushed in via :meth:`PaperBroker.update_marks` (the executor sets them from the latest
closed bar). Market orders fill immediately at mark ± slippage as taker; limit/POST-ONLY
orders fill at their limit price as maker when the mark reaches them — capturing the
bitbank maker rebate realistically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine.core.clock import utc_now
from engine.core.types import (
    AssetClass,
    Balance,
    Fill,
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
from engine.execution.broker import FEE_MODELS, Broker, FeeModel

_DEFAULT_QUOTE_BY_ASSET = {AssetClass.CRYPTO: "JPY", AssetClass.EQUITY: "USD"}


def _quote_currency(instrument: Instrument) -> str:
    return instrument.quote or _DEFAULT_QUOTE_BY_ASSET[instrument.asset_class]


@dataclass(slots=True)
class _Pos:
    qty: float = 0.0
    avg: float = 0.0


def _apply_fill(
    qty: float, avg: float, fill_qty_signed: float, price: float
) -> tuple[float, float, float]:
    """Update (qty, avg) for a signed fill; return (new_qty, new_avg, realized_pnl)."""
    new_qty = qty + fill_qty_signed
    realized = 0.0
    same_dir = qty == 0.0 or math.copysign(1, qty) == math.copysign(1, fill_qty_signed)
    if same_dir:
        denom = abs(qty) + abs(fill_qty_signed)
        new_avg = (abs(qty) * avg + abs(fill_qty_signed) * price) / denom if denom else 0.0
    else:
        closing = min(abs(fill_qty_signed), abs(qty))
        realized = closing * (price - avg) * math.copysign(1, qty)
        new_avg = price if abs(fill_qty_signed) > abs(qty) else avg
    if new_qty == 0.0:
        new_avg = 0.0
    return new_qty, new_avg, realized


class PaperBroker(Broker):
    def __init__(
        self,
        instruments: dict[str, Instrument],
        starting_cash: dict[str, float],
        venue: Venue = Venue.PAPER,
        fee_model: FeeModel | None = None,
        slippage_bps: float = 5.0,
    ) -> None:
        self.venue = venue
        self.instruments = dict(instruments)
        self.cash: dict[str, float] = dict(starting_cash)
        self.fee_model = fee_model or FEE_MODELS.get(venue, FeeModel(0.0, 0.0))
        self.slippage_bps = slippage_bps
        self._marks: dict[str, float] = {}
        self._positions: dict[str, _Pos] = {}
        self._open_orders: list[Order] = []
        self.fills: list[Fill] = []
        self.realized_pnl: dict[str, float] = {}

    # -- market data push ---------------------------------------------------------------

    def update_marks(self, marks: dict[str, float]) -> None:
        """Set current reference prices and fill any resting limit orders now marketable."""
        self._marks.update(marks)
        still_open: list[Order] = []
        for order in self._open_orders:
            if self._try_fill_limit(order):
                continue
            still_open.append(order)
        self._open_orders = still_open

    def mark(self, symbol: str) -> float | None:
        return self._marks.get(symbol)

    # -- Broker interface ---------------------------------------------------------------

    def get_balances(self) -> dict[str, Balance]:
        return {c: Balance(c, total=v, free=v, used=0.0) for c, v in self.cash.items()}

    def get_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for symbol, pos in self._positions.items():
            if pos.qty == 0.0:
                continue
            instrument = self.instruments[symbol]
            out[symbol] = Position(
                symbol=symbol,
                qty=pos.qty,
                avg_price=pos.avg,
                venue=self.venue,
                asset_class=instrument.asset_class,
            )
        return out

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        if symbol is None:
            return list(self._open_orders)
        return [o for o in self._open_orders if o.symbol == symbol]

    def submit_order(self, order: Order) -> Order:
        # Idempotency: a re-submitted client_order_id that already filled is a no-op.
        for fill in self.fills:
            if fill.client_order_id and fill.client_order_id == order.client_order_id:
                order.status = OrderStatus.FILLED
                return order
        if order.type is OrderType.MARKET:
            self._fill_market(order)
        else:
            if not self._try_fill_limit(order):
                order.status = OrderStatus.SUBMITTED
                self._open_orders.append(order)
        return order

    def cancel_order(self, broker_order_id: str, symbol: str | None = None) -> None:
        self._open_orders = [o for o in self._open_orders if o.client_order_id != broker_order_id]

    # -- helpers ------------------------------------------------------------------------

    def get_quote(self, symbol: str) -> Quote:
        m = self._marks.get(symbol, 0.0)
        return Quote(ts=utc_now(), bid=m, ask=m, last=m)

    def equity(self, marks: dict[str, float] | None = None) -> dict[str, float]:
        """Mark-to-market account value per quote currency."""
        marks = {**self._marks, **(marks or {})}
        equity = dict(self.cash)
        for symbol, pos in self._positions.items():
            if pos.qty == 0.0:
                continue
            currency = _quote_currency(self.instruments[symbol])
            equity[currency] = equity.get(currency, 0.0) + pos.qty * marks.get(symbol, pos.avg)
        return equity

    def snapshot(self) -> dict:
        """Serializable account state (cash + open positions) for committing to disk."""
        return {
            "cash": dict(self.cash),
            "positions": {
                s: {"qty": p.qty, "avg": p.avg} for s, p in self._positions.items() if p.qty != 0.0
            },
            "realized_pnl": dict(self.realized_pnl),
        }

    def apply_state(self, snapshot: dict) -> None:
        """Restore cash/positions from a previous :meth:`snapshot` (resume a paper run)."""
        if snapshot.get("cash"):
            self.cash = {k: float(v) for k, v in snapshot["cash"].items()}
        self._positions = {
            s: _Pos(qty=float(d["qty"]), avg=float(d["avg"]))
            for s, d in snapshot.get("positions", {}).items()
        }
        self.realized_pnl = {k: float(v) for k, v in snapshot.get("realized_pnl", {}).items()}

    def _require_mark(self, symbol: str) -> float:
        mark = self._marks.get(symbol)
        if mark is None or mark <= 0.0:
            raise ValueError(f"no mark price set for {symbol!r}; call update_marks() first")
        return mark

    def _fill_market(self, order: Order) -> None:
        mark = self._require_mark(order.symbol)
        price = mark * (1.0 + self.slippage_bps / 10_000.0 * order.side.sign)
        self._book_fill(order, price, is_maker=False)

    def _try_fill_limit(self, order: Order) -> bool:
        mark = self._marks.get(order.symbol)
        if mark is None or order.limit_price is None:
            return False
        marketable = (
            mark <= order.limit_price if order.side is Side.BUY else mark >= order.limit_price
        )
        if not marketable:
            return False
        is_maker = order.tif is TimeInForce.POST_ONLY or order.type is OrderType.LIMIT
        self._book_fill(order, order.limit_price, is_maker=is_maker)
        return True

    def _book_fill(self, order: Order, price: float, is_maker: bool) -> None:
        instrument = self.instruments[order.symbol]
        currency = _quote_currency(instrument)
        signed_qty = order.qty * order.side.sign
        notional = order.qty * price
        fee = self.fee_model.fee(notional, is_maker)

        # Cash flow: buy reduces cash by notional, sell increases; fee always subtracts.
        self.cash[currency] = self.cash.get(currency, 0.0) - order.side.sign * notional - fee

        pos = self._positions.setdefault(order.symbol, _Pos())
        new_qty, new_avg, realized = _apply_fill(pos.qty, pos.avg, signed_qty, price)
        pos.qty, pos.avg = new_qty, new_avg
        if realized:
            self.realized_pnl[order.symbol] = self.realized_pnl.get(order.symbol, 0.0) + realized

        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.avg_fill_price = price
        order.broker_order_id = order.client_order_id
        self.fills.append(
            Fill(
                ts=utc_now(),
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=price,
                fee=fee,
                fee_currency=currency,
                venue=self.venue,
                client_order_id=order.client_order_id,
                broker_order_id=order.broker_order_id,
            )
        )
