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


def test_research_only_records_can_never_be_promoted(tmp_path, monkeypatch):
    """A record validated on a non-venue series must never reach an executor.

    fred:DEXJPUS carries 55 years of USD/JPY, which is what makes trade_balance
    judgeable — but "fred" is not a venue. _resolve_instrument would fail to match the
    symbol, fall back to bitbank, and try to trade "DEXJPUS".
    """
    from engine.core.config import Settings, load_settings
    from engine.orchestration.population import Population, StrategyRecord
    from engine.orchestration.promotion import promote_step

    base = load_settings()
    settings = Settings(
        repo_root=base.repo_root,
        data_dir=base.data_dir,
        state_dir=tmp_path / "state",
        config_dir=base.config_dir,
        live_enabled=False,
        alpaca_paper=True,
        bitbank=base.bitbank,
        alpaca=base.alpaca,
    )
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    population = Population(settings.state_dir / "population.db")
    metrics = {"ppy": 252.0, "n": 13936, "sharpe_per_bar": 10.0, "research_only": True}
    population.upsert(
        StrategyRecord(
            family="trade_balance:DEXJPUS:1d",
            strategy="trade_balance",
            symbol="DEXJPUS",
            source="fred",
            timeframe="1d",
            params={},
            metrics=metrics,
            status="incubating",
            incubation_start_ts=0,
        )
    )
    population.close()

    summary = promote_step(settings)
    assert summary.n_promoted == 0, "a research-only record must never be promoted"

    population = Population(settings.state_dir / "population.db")
    assert population.get("trade_balance:DEXJPUS:1d")["status"] == "incubating"
    population.close()


def test_research_only_series_are_declared_and_kept_out_of_the_universe():
    """The deep series is researched, but is not an instrument anything can trade."""
    from engine.core.config import load_settings, load_universe

    universe = load_universe(load_settings().universe_path)
    assert any(r.symbol == "DEXJPUS" and r.source == "fred" for r in universe.research_series)
    assert all(e.instrument.symbol != "DEXJPUS" for e in universe.entries)
