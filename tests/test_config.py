from engine.core.config import _env_bool, load_settings, load_universe
from engine.core.types import AssetClass, Venue


def test_load_universe_from_repo():
    settings = load_settings()
    universe = load_universe(settings.universe_path)
    symbols = {e.instrument.symbol for e in universe.entries}
    assert {"BTC/JPY", "ETH/JPY", "XRP/JPY", "SPY", "QQQ", "AAPL"} <= symbols

    btc = next(e for e in universe.entries if e.instrument.symbol == "BTC/JPY")
    assert btc.instrument.asset_class is AssetClass.CRYPTO
    assert btc.instrument.venue is Venue.BITBANK
    assert btc.live_source.source == "bitbank"
    assert btc.research_source.source == "binance"
    assert btc.research_source.symbol == "BTC/USDT"
    assert btc.timeframes == ("1h", "4h", "1d")

    spy = next(e for e in universe.entries if e.instrument.symbol == "SPY")
    assert spy.instrument.asset_class is AssetClass.EQUITY
    assert spy.live_source.source == "alpaca"
    assert spy.research_source is None


def test_by_asset_class():
    settings = load_settings()
    universe = load_universe(settings.universe_path)
    assert len(universe.by_asset_class(AssetClass.CRYPTO)) == 3
    assert len(universe.by_asset_class(AssetClass.EQUITY)) == 3


def test_env_bool(monkeypatch):
    monkeypatch.setenv("FLAG", "true")
    assert _env_bool("FLAG") is True
    monkeypatch.setenv("FLAG", "0")
    assert _env_bool("FLAG") is False
    monkeypatch.delenv("FLAG", raising=False)
    assert _env_bool("FLAG", default=True) is True


def test_live_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TRADE_LIVE_ENABLED", raising=False)
    assert load_settings().live_enabled is False
