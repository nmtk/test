import numpy as np
import pandas as pd
import pytest

from engine.core.clock import utc_now
from engine.core.config import Settings, load_settings
from engine.core.killswitch import engage
from engine.core.types import AssetClass, Balance, Fill, OrderStatus, Position, Venue
from engine.data.store import OHLCVStore
from engine.execution.broker import Broker
from engine.orchestration.champions import Allocation, Champions, save_champions
from engine.orchestration.live import run_live_step


class FakeBroker(Broker):
    """In-memory live broker for tests: fills limit orders at their limit price."""

    venue = Venue.PAPER

    def __init__(self, cash: dict[str, float]):
        self._cash = dict(cash)
        self._pos: dict[str, tuple[float, float]] = {}
        self.fills: list = []

    def seed_position(self, symbol: str, qty: float, avg: float):
        self._pos[symbol] = (qty, avg)

    def get_balances(self):
        return {c: Balance(c, total=v, free=v) for c, v in self._cash.items()}

    def get_positions(self):
        return {
            s: Position(s, qty, avg, Venue.PAPER, AssetClass.CRYPTO)
            for s, (qty, avg) in self._pos.items()
            if qty != 0.0
        }

    def get_open_orders(self, symbol=None):
        return []

    def submit_order(self, order):
        price = order.limit_price
        if price is None:
            order.status = OrderStatus.REJECTED
            return order
        currency = next(iter(self._cash))
        self._cash[currency] -= order.side.sign * order.qty * price
        cur_qty, cur_avg = self._pos.get(order.symbol, (0.0, 0.0))
        signed = order.qty * order.side.sign
        new_qty = cur_qty + signed
        same_dir = cur_qty == 0.0 or (cur_qty > 0) == (signed > 0)
        if same_dir and new_qty != 0:
            avg = (abs(cur_qty) * cur_avg + abs(signed) * price) / (abs(cur_qty) + abs(signed))
        else:
            avg = price if abs(signed) > abs(cur_qty) else cur_avg
        self._pos[order.symbol] = (new_qty, 0.0 if new_qty == 0 else avg)
        order.status = OrderStatus.FILLED
        self.fills.append(
            Fill(
                utc_now(),
                order.symbol,
                order.side,
                order.qty,
                price,
                0.0,
                currency,
                Venue.PAPER,
                client_order_id=order.client_order_id,
            )
        )
        return order

    def cancel_order(self, broker_order_id, symbol=None):
        pass


def _settings(tmp_path, *, live: bool) -> Settings:
    base = load_settings()
    return Settings(
        repo_root=base.repo_root,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        config_dir=base.config_dir,
        live_enabled=live,
        alpaca_paper=True,
        bitbank=base.bitbank,
        alpaca=base.alpaca,
    )


def _seed(settings):
    n = 200
    close = 100.0 * (1.01 ** np.arange(n))  # uptrend -> tsmom long
    ts = np.arange(n) * 3_600_000 + 1_600_000_000_000
    df = pd.DataFrame(
        {"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 1.0}
    )
    OHLCVStore(settings.data_dir / "ohlcv").upsert("bitbank", "BTC/JPY", "1h", df)
    save_champions(
        settings.state_dir / "champions.json",
        Champions(
            [Allocation("tsmom", "BTC/JPY", "bitbank", "1h", 1.0, {"lookback": 20})], updated="t"
        ),
    )
    return float(close[-1])


def test_live_disabled_is_a_noop(tmp_path):
    settings = _settings(tmp_path, live=False)
    _seed(settings)
    called = {"n": 0}

    def factory(*a, **k):
        called["n"] += 1
        return FakeBroker({"JPY": 1_000_000.0})

    result = run_live_step(settings, broker_factory=factory, canary={Venue.BITBANK: 10_000.0})
    assert result.note == "live disabled"
    assert result.n_orders == 0
    assert called["n"] == 0  # never even built a broker


def test_canary_caps_deployed_capital(tmp_path):
    settings = _settings(tmp_path, live=True)
    price = _seed(settings)
    fake = FakeBroker({"JPY": 1_000_000.0})  # large balance...

    result = run_live_step(
        settings, broker_factory=lambda *a, **k: fake, canary={Venue.BITBANK: 10_000.0}
    )
    assert result.n_orders == 1
    qty = result.positions["BTC/JPY"]
    # deployed notional is capped near the canary (10k), NOT the full 1M balance
    assert qty * price == pytest.approx(10_000.0, rel=0.05)


def test_kill_switch_flattens_live(tmp_path):
    settings = _settings(tmp_path, live=True)
    price = _seed(settings)
    fake = FakeBroker({"JPY": 1_000_000.0})
    fake.seed_position("BTC/JPY", qty=10_000.0 / price, avg=price)
    engage(settings.state_dir, reason="test")

    result = run_live_step(
        settings, broker_factory=lambda *a, **k: fake, canary={Venue.BITBANK: 10_000.0}
    )
    assert result.killed is True
    assert result.positions == {}  # flattened
