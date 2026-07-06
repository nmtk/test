"""Stooq/FRED context sources (offline CSV parsing) and the context sync path."""

import numpy as np
import pandas as pd

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


def test_make_source_knows_context_sources():
    settings = load_settings()
    assert isinstance(make_source("stooq", settings), StooqSource)
    assert isinstance(make_source("fred", settings), FredSource)


def test_context_yaml_loads_series():
    settings = load_settings()
    series = load_context_series(settings.context_path)
    keys = {s.key for s in series}
    assert {"spx", "gold", "us10y", "usdjpy", "eurusd"} <= keys
    assert all(s.timeframe == "1d" for s in series)
    assert all(s.source in ("stooq", "fred") for s in series)


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
        config_dir=base.config_dir,  # real context.yaml (5 series)
        live_enabled=False,
        alpaca_paper=True,
        bitbank=base.bitbank,
        alpaca=base.alpaca,
    )
    stub = _StubSource()
    monkeypatch.setattr(sync_mod, "make_source", lambda source, s: stub)

    store = OHLCVStore(settings.data_dir / "ohlcv")
    first = sync_context(settings, store)
    assert len(first) == 5
    assert all(r.ok for r in first)
    assert all(r.rows_total == 5 for r in first)
    assert store.exists("stooq", "^spx", "1d")

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
