"""Live execution path (real money) — OFF by default, behind multiple hard locks.

This mirrors the paper executor but routes to real broker accounts per venue (bitbank for
crypto, Alpaca for equities) and reconciles against *real* positions. It refuses to do
anything unless ALL of these hold:

* ``settings.live_enabled`` is true (``TRADE_LIVE_ENABLED=true``), and the CI job is gated
  by the ``LIVE_ENABLED`` repository variable;
* trade-only API keys are present for the venue (created WITHOUT withdrawal permission);
* ``state/KILL_SWITCH`` does not exist (else it flattens and halts).

Even when enabled, deployment is capped by a small **canary** per-venue capital limit, so a
bug or a bad champion can only ever touch a tiny, staged amount — you raise the cap by hand.
"""

from __future__ import annotations

from collections import defaultdict

import engine.gp.strategy  # noqa: F401  -- registers the "gp" strategy for get_strategy
from engine.core.config import Settings, load_universe
from engine.core.killswitch import is_engaged
from engine.core.types import AssetClass, Venue
from engine.data.store import OHLCVStore
from engine.execution.broker import Broker, reconcile_orders
from engine.features.context import ContextAugmenter
from engine.orchestration import state as state_io
from engine.orchestration.champions import load_champions
from engine.orchestration.orchestrator import REBALANCE_BAND, StepResult, _resolve_instrument
from engine.orchestration.risk import RiskLimits, apply_limits
from engine.strategies import get_strategy

# Tiny default canary caps (deployable capital per venue, in the venue's quote currency).
# Staged rollout = raise these by hand once paper + canary look healthy.
DEFAULT_CANARY: dict[Venue, float] = {Venue.BITBANK: 10_000.0, Venue.ALPACA: 100.0}
_QUOTE_BY_VENUE: dict[Venue, str] = {Venue.BITBANK: "JPY", Venue.ALPACA: "USD"}


def _default_factory(venue: Venue, instruments: dict, settings: Settings) -> Broker | None:
    """Build a real broker for a venue, or None if its trade-only keys aren't configured."""
    if venue is Venue.BITBANK and settings.bitbank.is_configured:
        from engine.execution.brokers.bitbank import BitbankBroker

        return BitbankBroker(settings.bitbank.api_key, settings.bitbank.api_secret, instruments)
    if venue is Venue.ALPACA and settings.alpaca.is_configured:
        from engine.execution.brokers.alpaca import AlpacaBroker

        return AlpacaBroker(
            settings.alpaca.api_key, settings.alpaca.api_secret, paper=settings.alpaca_paper
        )
    return None


def run_live_step(
    settings: Settings,
    *,
    broker_factory=None,
    canary: dict[Venue, float] | None = None,
    limits: RiskLimits | None = None,
) -> StepResult:
    """One live execution step. Returns immediately (no trading) unless live is enabled."""
    if not settings.live_enabled:
        return StepResult(mode="live", bar_ts_ms=0, n_orders=0, killed=False, note="live disabled")

    canary = canary or DEFAULT_CANARY
    limits = limits or RiskLimits()
    factory = broker_factory or _default_factory
    killed = is_engaged(settings.state_dir)

    champions = load_champions(settings.state_dir / "champions.json")
    universe = load_universe(settings.universe_path)
    store = OHLCVStore(settings.data_dir / "ohlcv")
    augmenter = ContextAugmenter.from_settings(settings, universe)
    instruments = {
        a.symbol: _resolve_instrument(a.symbol, a.source, universe) for a in champions.allocations
    }

    by_venue: dict[Venue, list] = defaultdict(list)
    for alloc in champions.allocations:
        by_venue[instruments[alloc.symbol].venue].append(alloc)

    total_orders = 0
    latest_ts = 0
    equity_all: dict[str, float] = {}
    positions_all: dict[str, float] = {}
    new_fills: list = []

    for venue, allocs in by_venue.items():
        v_instruments = {a.symbol: instruments[a.symbol] for a in allocs}
        broker = factory(venue, v_instruments, settings)
        if broker is None:  # keys not configured for this venue
            continue

        marks, ts, weights = _marks_and_weights(store, allocs, augmenter)
        if not marks:
            continue
        latest_ts = max(latest_ts, ts)
        weights = (
            dict.fromkeys(marks, 0.0) if killed else apply_limits(weights, v_instruments, limits)
        )

        quote = _QUOTE_BY_VENUE.get(venue, "USD")
        current = broker.get_positions()
        book = _canary_equity(broker, current, marks, quote, canary.get(venue, 0.0))

        targets = _size_targets(weights, marks, current, book)
        orders = reconcile_orders(
            current,
            targets,
            v_instruments,
            bar_ts_ms=ts,
            prefix="live",
            limit_prices=marks,
            ref_prices=marks,
        )
        for order in orders:
            broker.submit_order(order)
            total_orders += 1
        new_fills.extend(getattr(broker, "fills", []))

        for cur, bal in broker.get_balances().items():
            equity_all[cur] = equity_all.get(cur, 0.0) + bal.total
        positions_all.update({s: p.qty for s, p in broker.get_positions().items()})

    state_io.append_fills(state_io.fills_path(settings.state_dir, "live"), new_fills, latest_ts)
    if equity_all:
        state_io.append_equity(
            state_io.equity_path(settings.state_dir, "live"), latest_ts, equity_all
        )

    note = "flattened (kill switch)" if killed else ("ok" if total_orders else "no trades")
    return StepResult(
        mode="live",
        bar_ts_ms=latest_ts,
        n_orders=total_orders,
        killed=killed,
        equity=equity_all,
        positions=positions_all,
        note=note,
    )


def _marks_and_weights(store: OHLCVStore, allocs: list, augmenter: ContextAugmenter):
    marks: dict[str, float] = {}
    weights: dict[str, float] = defaultdict(float)
    latest_ts = 0
    for alloc in allocs:
        bars = store.read(alloc.source, alloc.symbol, alloc.timeframe)
        if len(bars) < 2:
            continue
        marks[alloc.symbol] = float(bars["close"].iloc[-1])
        latest_ts = max(latest_ts, int(bars["ts"].iloc[-1]))
        bars = augmenter.augment(
            bars, source=alloc.source, symbol=alloc.symbol, timeframe=alloc.timeframe
        )
        w = float(get_strategy(alloc.strategy)(**alloc.params).weights(bars).iloc[-1])
        weights[alloc.symbol] += alloc.weight * w
    return marks, latest_ts, dict(weights)


def _canary_equity(broker, current, marks, quote: str, cap: float) -> float:
    """Deployable capital = min(canary cap, real account equity in the quote currency)."""
    balances = broker.get_balances()
    cash = balances[quote].total if quote in balances else 0.0
    pos_value = sum(p.qty * marks.get(s, 0.0) for s, p in current.items())
    equity = cash + pos_value
    return min(cap, equity) if cap > 0 else 0.0


def _size_targets(weights, marks, current, book: float) -> dict[str, float]:
    targets: dict[str, float] = {}
    for symbol, weight in weights.items():
        price = marks[symbol]
        desired = (weight * book / price) if price > 0 else 0.0
        current_qty = current[symbol].qty if symbol in current else 0.0
        if abs(desired - current_qty) * price < REBALANCE_BAND * max(book, 1.0):
            desired = current_qty  # no-trade band (anti-churn / idempotency)
        targets[symbol] = desired
    return targets


# AssetClass imported for type-completeness of instrument resolution callers.
_ = AssetClass
