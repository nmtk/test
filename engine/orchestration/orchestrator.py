"""The executor step: champions -> target weights -> risk -> orders -> fills -> state.

One ``run_step`` serves both paper and live. It is idempotent and self-reconciling: it sizes
*target* positions from the latest closed bars and trades only the difference, so a delayed,
duplicated, or resumed cron run converges rather than double-trading. The kill switch, if
engaged, forces every target to zero before any order is placed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import engine.gp.strategy  # noqa: F401  -- registers the "gp" strategy for get_strategy
from engine.core.config import Settings, load_universe
from engine.core.killswitch import is_engaged
from engine.core.types import AssetClass, Instrument, Venue
from engine.data.store import OHLCVStore
from engine.execution.broker import FeeModel, reconcile_orders
from engine.execution.paper import PaperBroker
from engine.features.context import ContextAugmenter
from engine.orchestration import state as state_io
from engine.orchestration.champions import load_champions
from engine.orchestration.risk import RiskLimits, apply_limits
from engine.strategies import get_strategy

# Paper books start here; conservative cost model (bitbank taker + slippage) so paper P&L
# is not rosier than reality during incubation.
PAPER_INITIAL_CASH: dict[str, float] = {"JPY": 1_000_000.0, "USD": 10_000.0}
PAPER_FEE = FeeModel(maker_bps=0.0, taker_bps=12.0)
PAPER_SLIPPAGE_BPS = 5.0
# Skip rebalances smaller than this fraction of book equity (anti-churn / idempotency).
REBALANCE_BAND = 0.02


@dataclass(slots=True)
class StepResult:
    mode: str
    bar_ts_ms: int
    n_orders: int
    killed: bool
    equity: dict[str, float] = field(default_factory=dict)
    positions: dict[str, float] = field(default_factory=dict)
    note: str = ""


def _resolve_instrument(symbol: str, source: str, universe) -> Instrument:
    for entry in universe.entries:
        if entry.instrument.symbol == symbol:
            return entry.instrument
    asset = AssetClass.CRYPTO if "/" in symbol else AssetClass.EQUITY
    try:
        venue = Venue(source)
    except ValueError:
        venue = Venue.BITBANK if asset is AssetClass.CRYPTO else Venue.ALPACA
    base = symbol.split("/")[0] if "/" in symbol else None
    quote = symbol.split("/")[1] if "/" in symbol else "USD"
    return Instrument(symbol=symbol, asset_class=asset, venue=venue, base=base, quote=quote)


def run_step(
    settings: Settings,
    *,
    mode: str = "paper",
    initial_cash: dict[str, float] | None = None,
    limits: RiskLimits | None = None,
) -> StepResult:
    limits = limits or RiskLimits()
    champions = load_champions(settings.state_dir / "champions.json")
    universe = load_universe(settings.universe_path)
    store = OHLCVStore(settings.data_dir / "ohlcv")

    instruments = {
        a.symbol: _resolve_instrument(a.symbol, a.source, universe) for a in champions.allocations
    }
    broker = PaperBroker(
        instruments,
        dict(initial_cash or PAPER_INITIAL_CASH),
        venue=Venue.PAPER,
        fee_model=PAPER_FEE,
        slippage_bps=PAPER_SLIPPAGE_BPS,
    )
    acct_path = state_io.account_path(settings.state_dir, mode)
    broker.apply_state(state_io.load_account(acct_path))

    # Latest closed-bar marks + aggregated target weight per symbol. Bars get the same
    # ctx_* context columns research validated on, so champions see identical features.
    augmenter = ContextAugmenter.from_settings(settings, universe)
    marks: dict[str, float] = {}
    latest_ts = 0
    target_weights: dict[str, float] = defaultdict(float)
    for alloc in champions.allocations:
        bars = store.read(alloc.source, alloc.symbol, alloc.timeframe)
        if len(bars) < 2:
            continue
        marks[alloc.symbol] = float(bars["close"].iloc[-1])
        latest_ts = max(latest_ts, int(bars["ts"].iloc[-1]))
        bars = augmenter.augment(
            bars, source=alloc.source, symbol=alloc.symbol, timeframe=alloc.timeframe
        )
        weight = float(get_strategy(alloc.strategy)(**alloc.params).weights(bars).iloc[-1])
        target_weights[alloc.symbol] += alloc.weight * weight

    if not marks:
        return _finalize(
            broker, settings, mode, latest_ts, n_orders=0, killed=False, note="no data"
        )

    broker.update_marks(marks)
    killed = is_engaged(settings.state_dir)
    if killed:
        target_weights = dict.fromkeys(marks, 0.0)  # flatten everything
    else:
        target_weights = apply_limits(dict(target_weights), instruments, limits)

    # Size target weights into target quantities against each symbol's quote-currency equity.
    equity = broker.equity()
    current = broker.get_positions()
    targets_qty: dict[str, float] = {}
    for symbol, weight in target_weights.items():
        instrument = instruments[symbol]
        currency = instrument.quote or (
            "JPY" if instrument.asset_class is AssetClass.CRYPTO else "USD"
        )
        price = marks[symbol]
        book_equity = equity.get(currency, 0.0)
        desired = (weight * book_equity / price) if price > 0 else 0.0
        current_qty = current[symbol].qty if symbol in current else 0.0
        # No-trade band: ignore dust rebalances (keeps cost down and makes re-running the
        # same bar idempotent, since cost-driven equity drift won't trigger a tiny trade).
        if abs(desired - current_qty) * price < REBALANCE_BAND * book_equity:
            desired = current_qty
        targets_qty[symbol] = desired

    orders = reconcile_orders(
        current, targets_qty, instruments, bar_ts_ms=latest_ts, prefix=mode, ref_prices=marks
    )
    for order in orders:
        broker.submit_order(order)

    note = "flattened (kill switch)" if killed else "ok"
    return _finalize(
        broker, settings, mode, latest_ts, n_orders=len(orders), killed=killed, note=note
    )


def _finalize(
    broker: PaperBroker,
    settings: Settings,
    mode: str,
    bar_ts_ms: int,
    *,
    n_orders: int,
    killed: bool,
    note: str,
) -> StepResult:
    state_io.save_account(state_io.account_path(settings.state_dir, mode), broker)
    state_io.append_fills(state_io.fills_path(settings.state_dir, mode), broker.fills, bar_ts_ms)
    equity = broker.equity()
    state_io.append_equity(state_io.equity_path(settings.state_dir, mode), bar_ts_ms, equity)
    return StepResult(
        mode=mode,
        bar_ts_ms=bar_ts_ms,
        n_orders=n_orders,
        killed=killed,
        equity=equity,
        positions={s: p.qty for s, p in broker.get_positions().items()},
        note=note,
    )
