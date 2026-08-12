"""Stooq/FRED context sources (offline CSV parsing) and the context sync path."""

import numpy as np
import pandas as pd
import pytest

import engine.data.sync as sync_mod
from engine.core.config import Settings, load_context_series, load_settings
from engine.data.schema import OHLCV_COLUMNS
from engine.data.sources import FredSource, StooqSource, make_source
from engine.data.store import OHLCVStore
from engine.data.sync import sync_context

STOOQ_CSV = """Date,Open,High,Low,Close,Volume
2026-06-29,6100.5,6150.0,6080.0,6141.0,1000
2026-06-30,6141.0,6160.0,6120.0,6155.3,1100
2026-07-01,6155.3,6180.0,6150.0,6170.1,900
"""

STOOQ_CSV_NO_VOLUME = """Date,Open,High,Low,Close
2026-06-30,1.157,1.160,1.152,1.158
2026-07-01,1.158,1.162,1.155,1.161
"""

FRED_CSV = """DATE,DGS10
2026-06-29,4.28
2026-06-30,.
2026-07-01,4.31
"""


def test_stooq_parse_canonical():
    df = StooqSource._parse_csv(STOOQ_CSV)
    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 3
    # 2026-06-29 UTC midnight in epoch ms
    assert df["ts"].iloc[0] == int(pd.Timestamp("2026-06-29", tz="UTC").value // 1_000_000)
    assert df["close"].iloc[-1] == 6170.1
    assert df["volume"].iloc[0] == 1000


def test_stooq_parse_without_volume_defaults_zero():
    df = StooqSource._parse_csv(STOOQ_CSV_NO_VOLUME)
    assert len(df) == 2
    assert (df["volume"] == 0.0).all()


def test_stooq_parse_garbage_is_empty():
    assert len(StooqSource._parse_csv("No data")) == 0


def test_fred_parse_drops_missing_prints():
    df = FredSource._parse_csv(FRED_CSV, "DGS10")
    assert len(df) == 2  # the "." row is dropped
    assert (df["open"] == df["close"]).all()
    assert (df["volume"] == 0.0).all()
    assert df["close"].tolist() == [4.28, 4.31]


def test_stooq_fetch_sends_browser_user_agent(monkeypatch):
    # stooq 404s the default python-requests agent (observed on every Actions run),
    # so the fetch must present a browser User-Agent.
    captured = {}

    class _Resp:
        text = STOOQ_CSV

        def raise_for_status(self):
            pass

    import requests

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers or {}
        return _Resp()

    monkeypatch.setattr(requests, "get", fake_get)
    df = StooqSource().fetch_ohlcv("xauusd", "1d")
    assert len(df) == 3
    assert "Mozilla" in captured["headers"].get("User-Agent", "")


def test_make_source_knows_context_sources():
    settings = load_settings()
    assert isinstance(make_source("stooq", settings), StooqSource)
    assert isinstance(make_source("fred", settings), FredSource)


def test_context_yaml_loads_series():
    settings = load_settings()
    series = load_context_series(settings.context_path)
    keys = {s.key for s in series}
    assert {"spx", "gold", "us10y", "usdjpy", "eurusd"} <= keys
    assert all(s.source in ("stooq", "fred") for s in series)
    # Market prices are daily and known instantly; released statistics are monthly and
    # must declare a publication lag (see engine/features/macro_trade.py).
    for spec in series:
        if spec.timeframe == "1d":
            assert spec.release_lag_days is None, f"{spec.key} is a price, not a release"
        else:
            assert spec.timeframe == "1mo"
            assert spec.release_lag_days and spec.release_lag_days > 0, spec.key
    # FRED is primary (Actions-reachable): only gold, which FRED lacks, stays on stooq.
    by_key = {s.key: s for s in series}
    assert by_key["spx"].source == "fred" and by_key["spx"].symbol == "SP500"
    assert by_key["us10y"].source == "fred" and by_key["us10y"].symbol == "DGS10"
    assert by_key["usdjpy"].source == "fred" and by_key["usdjpy"].symbol == "DEXJPUS"
    assert by_key["eurusd"].source == "fred" and by_key["eurusd"].symbol == "DEXUSEU"
    assert by_key["gold"].source == "stooq"


class _StubSource:
    """Offline stand-in: serves a fixed daily series regardless of symbol."""

    def __init__(self):
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since_ms=None, limit=None):
        self.calls += 1
        ts = np.arange(5, dtype="int64") * 86_400_000 + 1_700_000_000_000
        close = np.linspace(100.0, 104.0, 5)
        df = pd.DataFrame(
            {"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 0.0}
        )
        if since_ms is not None:
            df = df[df["ts"] >= since_ms]
        return df


def test_sync_context_writes_and_is_idempotent(tmp_path, monkeypatch):
    base = load_settings()
    settings = Settings(
        repo_root=base.repo_root,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        config_dir=base.config_dir,  # the real context.yaml
        live_enabled=False,
        alpaca_paper=True,
        bitbank=base.bitbank,
        alpaca=base.alpaca,
    )
    stub = _StubSource()
    monkeypatch.setattr(sync_mod, "make_source", lambda source, s: stub)

    store = OHLCVStore(settings.data_dir / "ohlcv")
    configured = load_context_series(settings.context_path)
    first = sync_context(settings, store)
    assert len(first) == len(configured)
    assert all(r.ok for r in first)
    assert all(r.rows_total == 5 for r in first)
    # FRED is the primary source (stooq 404s GitHub Actions runners); gold stays stooq.
    assert store.exists("fred", "SP500", "1d")
    assert store.exists("stooq", "xauusd", "1d")
    # Released statistics land under their own timeframe key, not mixed in with prices.
    assert store.exists("fred", "EXPJP", "1mo")

    second = sync_context(settings, store)  # re-run converges, never duplicates
    assert all(r.ok and r.rows_total == 5 for r in second)


def test_sync_context_without_config_is_noop(tmp_path):
    base = load_settings()
    settings = Settings(
        repo_root=base.repo_root,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        config_dir=tmp_path / "empty_config",  # no context.yaml here
        live_enabled=False,
        alpaca_paper=True,
        bitbank=base.bitbank,
        alpaca=base.alpaca,
    )
    store = OHLCVStore(settings.data_dir / "ohlcv")
    assert sync_context(settings, store) == []


def test_fred_stores_monthly_release_series_not_only_daily():
    """The trade-flow series are monthly. Rejecting "1mo" silently disabled all four.

    The first deployment failed here: context.yaml labelled them 1mo, FredSource still
    hard-required 1d, and every trade-flow sync raised before a single byte was fetched.
    """
    source = FredSource()
    assert "1mo" in source.SUPPORTED_TIMEFRAMES
    assert "1d" in source.SUPPORTED_TIMEFRAMES
    with pytest.raises(ValueError, match="fred source stores"):
        source.fetch_ohlcv("EXPJP", "4h")


def test_every_configured_context_timeframe_is_supported_by_its_source():
    """Guards the class of bug above: config and source must agree, for every series."""
    settings = load_settings()
    for spec in load_context_series(settings.context_path):
        source = make_source(spec.source, settings)
        supported = getattr(source, "SUPPORTED_TIMEFRAMES", ("1d",))
        assert spec.timeframe in supported, (
            f"{spec.key} is configured as {spec.timeframe!r} but "
            f"{spec.source} only stores {supported}"
        )


class _DeepSource:
    """Serves a long monthly history, and records what window was asked for."""

    def __init__(self):
        self.since_values: list[int | None] = []

    def fetch_ohlcv(self, symbol, timeframe, since_ms=None, limit=None):
        self.since_values.append(since_ms)
        n = 300  # ~25 years of monthly prints
        idx = pd.date_range("2000-01-01", periods=n, freq="MS", tz="UTC")
        ts = idx.astype("datetime64[ms, UTC]").astype("int64").to_numpy()
        close = np.linspace(10.0, 60.0, n)
        df = pd.DataFrame(
            {"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 0.0}
        )
        return df[df["ts"] >= since_ms] if since_ms is not None else df


def test_monthly_series_are_fetched_whole_not_incrementally(tmp_path):
    """The 3-year seed capped trade-flow at 34 prints; the features need 36 to turn on."""
    from engine.core.config import DataSourceSpec
    from engine.data.sync import sync_series

    store = OHLCVStore(tmp_path / "ohlcv")
    source = _DeepSource()
    spec = DataSourceSpec(source="fred", symbol="EXPJP")

    first = sync_series(store, source, spec, "1mo")
    assert first.ok and first.rows_total == 300, "a monthly series must arrive in full"
    assert source.since_values == [None], "no seed window may be applied to releases"

    # A second run must still ask for everything, so revisions to earlier months land.
    sync_series(store, source, spec, "1mo")
    assert source.since_values == [None, None]


def test_daily_bars_still_sync_incrementally(tmp_path):
    """Price bars keep the incremental path — re-fetching years of them every run is waste."""
    from engine.core.config import DataSourceSpec
    from engine.data.sync import sync_series

    store = OHLCVStore(tmp_path / "ohlcv")
    source = _DeepSource()
    spec = DataSourceSpec(source="fred", symbol="DEXJPUS")

    sync_series(store, source, spec, "1d")
    assert source.since_values[0] is not None, "bars should be seeded, not fetched whole"
    sync_series(store, source, spec, "1d")
    assert source.since_values[1] is not None and source.since_values[1] > source.since_values[0]


def test_enough_history_makes_trade_features_non_neutral(tmp_path):
    """End to end: a full-depth sync must actually turn the trade_balance signal on."""
    from engine.core.config import DataSourceSpec
    from engine.data.sync import sync_series
    from engine.features.macro_trade import build_trade_features

    store = OHLCVStore(tmp_path / "ohlcv")
    for symbol in ("EXPJP", "IMPJP", "XTEXVA01JPM667S", "XTIMVA01JPM667S"):
        sync_series(store, _DeepSource(), DataSourceSpec(source="fred", symbol=symbol), "1mo")

    frames = {
        key: store.read("fred", symbol, "1mo")
        for key, symbol in (
            ("us_exports", "EXPJP"),
            ("us_imports", "IMPJP"),
            ("jp_exports", "XTEXVA01JPM667S"),
            ("jp_imports", "XTIMVA01JPM667S"),
        )
    }
    assert all(len(f) == 300 for f in frames.values())

    ts = pd.date_range("2010-01-01", periods=4000, freq="D", tz="UTC")
    bars = pd.DataFrame({"ts": ts.astype("datetime64[ms, UTC]").astype("int64"), "close": 150.0})
    bars = bars.assign(open=150.0, high=150.0, low=150.0, volume=0.0)
    features = build_trade_features(bars, timeframe="1d", frames=frames)
    assert not bool((features == 0.0).all().all()), "features must not stay inert with real depth"
