import pandas as pd

from engine.core.types import AssetClass, Instrument, Order, OrderType, Side, Venue
from engine.execution.paper import PaperBroker
from engine.orchestration import state as state_io

BTC = Instrument("BTC/JPY", AssetClass.CRYPTO, Venue.BITBANK, base="BTC", quote="JPY")


def _broker_with_position():
    pb = PaperBroker({"BTC/JPY": BTC}, {"JPY": 1_000_000.0}, venue=Venue.PAPER, slippage_bps=0)
    pb.update_marks({"BTC/JPY": 5_000_000.0})
    pb.submit_order(Order("o1", "BTC/JPY", Side.BUY, 0.1, type=OrderType.MARKET))
    return pb


def test_paper_broker_snapshot_roundtrip():
    pb = _broker_with_position()
    snap = pb.snapshot()

    restored = PaperBroker({"BTC/JPY": BTC}, {"JPY": 0.0}, venue=Venue.PAPER)
    restored.apply_state(snap)
    assert restored.cash["JPY"] == pb.cash["JPY"]
    assert restored.get_positions()["BTC/JPY"].qty == pb.get_positions()["BTC/JPY"].qty


def test_account_save_load(tmp_path):
    pb = _broker_with_position()
    path = state_io.account_path(tmp_path, "paper")
    state_io.save_account(path, pb)
    loaded = state_io.load_account(path)
    assert loaded["positions"]["BTC/JPY"]["qty"] == pb.get_positions()["BTC/JPY"].qty


def test_append_fills_and_equity(tmp_path):
    pb = _broker_with_position()
    fpath = state_io.fills_path(tmp_path, "paper")
    state_io.append_fills(fpath, pb.fills, ts_ms=1000)
    state_io.append_fills(fpath, pb.fills, ts_ms=2000)  # append again
    fills = pd.read_parquet(fpath)
    assert len(fills) == 2
    assert set(fills["symbol"]) == {"BTC/JPY"}

    epath = state_io.equity_path(tmp_path, "paper")
    state_io.append_equity(epath, 1000, pb.equity())
    state_io.append_equity(epath, 2000, pb.equity())
    eq = pd.read_parquet(epath)
    assert len(eq) == 2
    assert "JPY" in set(eq["currency"])
