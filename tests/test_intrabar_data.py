"""Intrabar bid/ask path: aggregation, storage, sync, and attaching it to bars."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from engine.core.config import Settings, load_settings, load_universe
from engine.data.dukascopy import DukascopySource, ticks_to_ohlcv
from engine.data.intrabar import (
    BACKFILL_MAX_HOURS,
    MIN_COVERAGE,
    PATH_LABEL,
    STEP_MS,
    IntrabarPath,
    IntrabarStore,
    attach_intrabar,
    coverage,
    intrabar_of,
    pip_size,
    prepare_intrabar_bars,
    sync_intrabar,
    ticks_to_path,
)
from engine.data.store import OHLCVStore

MIN = 60_000
T0 = 1_704_067_200_000  # 2024-01-01T00:00Z


def _ticks(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts", "bid", "ask"])


# -- aggregation ----------------------------------------------------------------------------


def test_pip_is_ten_feed_points():
    assert pip_size("USD/JPY") == pytest.approx(0.01)
    assert pip_size("EUR/USD") == pytest.approx(0.0001)


def test_path_keeps_both_sides_ohlc_per_minute():
    ticks = _ticks(
        [
            (T0 + 1_000, 150.000, 150.003),
            (T0 + 20_000, 150.010, 150.012),
            (T0 + 40_000, 149.990, 149.995),
            (T0 + 59_000, 150.004, 150.006),
            (T0 + MIN + 5_000, 150.020, 150.030),
        ]
    )
    path = IntrabarPath.from_frame(ticks_to_path(ticks, "USD/JPY"), "USD/JPY")
    assert list(path.ts) == [T0, T0 + MIN]
    assert path.bid_open[0] == pytest.approx(150.000)
    assert path.bid_high[0] == pytest.approx(150.010)
    assert path.bid_low[0] == pytest.approx(149.990)
    assert path.bid_close[0] == pytest.approx(150.004)
    assert path.ask_open[0] == pytest.approx(150.003)
    assert path.ask_high[0] == pytest.approx(150.012)
    assert path.ask_low[0] == pytest.approx(149.995)
    assert path.ask_close[0] == pytest.approx(150.006)
    assert path.bid_open[1] == pytest.approx(150.020)
    assert path.ask_open[1] == pytest.approx(150.030)


def test_new_extreme_steps_are_measured_inside_the_minute_only():
    """Only steps that set a new low (bid) / high (ask) count; the open gap does not."""
    ticks = _ticks(
        [
            (T0 + 1_000, 150.000, 150.002),
            (T0 + 2_000, 149.995, 149.997),  # bid new low by 5; ask falls (not a high)
            (T0 + 3_000, 150.008, 150.010),  # ask new high by 8 over 150.002
            (T0 + 4_000, 149.993, 149.995),  # bid falls 15, but undercuts the low by 2
            (T0 + MIN + 1_000, 149.900, 149.902),  # -93 across the boundary: a gap
            (T0 + MIN + 2_000, 149.899, 149.901),
        ]
    )
    path = IntrabarPath.from_frame(ticks_to_path(ticks, "USD/JPY"), "USD/JPY")
    assert path.bid_low_step[0] == pytest.approx(0.005)
    assert path.ask_high_step[0] == pytest.approx(0.008)
    assert path.bid_low_step[1] == pytest.approx(0.001)
    assert path.ask_high_step[1] == pytest.approx(0.0)


def test_path_round_trips_exactly_on_a_random_stream(make_ticks):
    ticks = make_ticks(hours=3, seed=3)
    path = IntrabarPath.from_frame(ticks_to_path(ticks, "USD/JPY"), "USD/JPY")
    minute = (ticks["ts"] // MIN) * MIN
    g = ticks.groupby(minute)
    np.testing.assert_array_equal(path.ts, g["bid"].first().index.to_numpy())
    np.testing.assert_allclose(path.bid_high, g["bid"].max().to_numpy(), atol=1e-9)
    np.testing.assert_allclose(path.bid_low, g["bid"].min().to_numpy(), atol=1e-9)
    np.testing.assert_allclose(path.ask_high, g["ask"].max().to_numpy(), atol=1e-9)
    np.testing.assert_allclose(path.ask_low, g["ask"].min().to_numpy(), atol=1e-9)
    np.testing.assert_allclose(path.ask_close, g["ask"].last().to_numpy(), atol=1e-9)


def test_empty_ticks_make_an_empty_path():
    frame = ticks_to_path(_ticks([]), "USD/JPY")
    assert len(frame) == 0
    assert len(IntrabarPath.from_frame(frame, "USD/JPY")) == 0


def test_path_is_immutable_and_never_copied_by_pandas_attrs(make_ticks):
    path = IntrabarPath.from_frame(ticks_to_path(make_ticks(hours=1), "USD/JPY"), "USD/JPY")
    with pytest.raises(ValueError):
        path.bid_open[0] = 0.0
    assert copy.deepcopy(path) is path

    bars = pd.DataFrame({"ts": path.ts[::5], "close": path.bid_close[::5]})
    attached = attach_intrabar(bars, path, 5 * MIN)
    # Column access and filtering propagate attrs; the path must ride along by reference.
    assert intrabar_of(attached[["close"]])[0] is path
    assert intrabar_of(attached[attached["close"] > 0])[0] is path


# -- storage --------------------------------------------------------------------------------


def test_store_round_trip_and_monthly_partitions(tmp_path, make_ticks):
    ticks = make_ticks(hours=4, start_ms=1_706_738_400_000)  # 2024-01-31T22:00Z, spans Feb
    frame = ticks_to_path(ticks, "USD/JPY")
    store = IntrabarStore(tmp_path)
    assert store.upsert("dukascopy", "USD/JPY", frame) == len(frame)
    names = [p.name for p in store.files("dukascopy", "USD/JPY")]
    assert names == ["2024-01.parquet", "2024-02.parquet"]
    pd.testing.assert_frame_equal(store.read_frame("dukascopy", "USD/JPY"), frame)
    assert store.last_ts("dukascopy", "USD/JPY") == int(frame["ts"].iloc[-1])


def test_store_upsert_is_idempotent_and_overwrites_overlap(tmp_path, make_ticks):
    frame = ticks_to_path(make_ticks(hours=2), "USD/JPY")
    store = IntrabarStore(tmp_path)
    store.upsert("dukascopy", "USD/JPY", frame)
    assert store.upsert("dukascopy", "USD/JPY", frame) == len(frame)

    revised = frame.tail(3).copy()
    revised["bid_close_d"] = 7
    assert store.upsert("dukascopy", "USD/JPY", revised) == len(frame)
    stored = store.read_frame("dukascopy", "USD/JPY")
    assert (stored["bid_close_d"].tail(3) == 7).all()


def test_store_on_an_empty_series():
    store = IntrabarStore("/nonexistent/intrabar")
    assert store.last_ts("dukascopy", "USD/JPY") is None
    assert len(store.read("dukascopy", "USD/JPY")) == 0


# -- coverage and attaching -------------------------------------------------------------------


def test_coverage_counts_bars_with_path_inside_them():
    path = IntrabarPath.from_frame(
        ticks_to_path(_ticks([(T0 + i * MIN + 1, 150.0, 150.002) for i in range(10)]), "USD/JPY"),
        "USD/JPY",
    )
    bar_ts = np.array([T0, T0 + 5 * MIN, T0 + 10 * MIN, T0 + 15 * MIN])
    assert coverage(bar_ts, 5 * MIN, path) == pytest.approx(0.5)


def test_attach_keeps_only_the_bars_the_path_covers(make_ticks):
    ticks = make_ticks(hours=6)
    bars = ticks_to_ohlcv(ticks, "5m")
    path = IntrabarPath.from_frame(
        ticks_to_path(ticks.iloc[len(ticks) // 2 :], "USD/JPY"), "USD/JPY"
    )
    attached = attach_intrabar(bars, path, 5 * MIN)
    assert len(attached) < len(bars)
    assert attached["ts"].iloc[0] >= path.ts[0]
    assert attached["ts"].iloc[-1] + 5 * MIN <= path.end_ts


def test_prepare_refuses_a_partially_backfilled_path(tmp_path, make_ticks):
    ticks = make_ticks(hours=10)
    bars = ticks_to_ohlcv(ticks, "5m")
    store = IntrabarStore(tmp_path)
    store.upsert("dukascopy", "USD/JPY", ticks_to_path(ticks.iloc[: len(ticks) // 2], "USD/JPY"))

    prepared, cov = prepare_intrabar_bars(
        bars, store, source="dukascopy", symbol="USD/JPY", timeframe="5m"
    )
    assert prepared is None
    assert cov < MIN_COVERAGE

    store.upsert("dukascopy", "USD/JPY", ticks_to_path(ticks, "USD/JPY"))
    prepared, cov = prepare_intrabar_bars(
        bars, store, source="dukascopy", symbol="USD/JPY", timeframe="5m"
    )
    assert prepared is not None and cov >= MIN_COVERAGE
    assert intrabar_of(prepared)[1] == 5 * MIN


def test_prepare_needs_a_two_sided_source_and_a_whole_number_of_minutes(tmp_path, make_ticks):
    ticks = make_ticks(hours=2)
    store = IntrabarStore(tmp_path)
    store.upsert("dukascopy", "USD/JPY", ticks_to_path(ticks, "USD/JPY"))
    bars = ticks_to_ohlcv(ticks, "5m")

    def prepared(source, timeframe):
        frame = bars if timeframe == "5m" else ticks_to_ohlcv(ticks, timeframe)
        return prepare_intrabar_bars(
            frame, store, source=source, symbol="USD/JPY", timeframe=timeframe
        )[0]

    assert prepared("dukascopy", "5m") is not None
    assert prepared("bitbank", "5m") is None, "a mid-only feed has no bid/ask path"
    assert STEP_MS == MIN
    assert prepared("dukascopy", "30s") is None, "a bar must hold whole path minutes"


# -- sync -----------------------------------------------------------------------------------


def _tmp_settings(tmp_path) -> Settings:
    base = load_settings()
    return Settings(
        repo_root=base.repo_root,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        config_dir=base.config_dir,
        live_enabled=False,
        alpaca_paper=True,
        bitbank=base.bitbank,
        alpaca=base.alpaca,
    )


class _FakeFeed:
    def __init__(self, ticks: pd.DataFrame, fail: bool = False) -> None:
        self.ticks = ticks
        self.fail = fail
        self.calls: list[int | None] = []
        self.budgets: list[int | None] = []

    def fetch_ticks(self, symbol, since_ms=None, max_hours=None):
        self.calls.append(since_ms)
        self.budgets.append(max_hours)
        if self.fail:
            raise OSError("connect timeout")
        return self.ticks[self.ticks["ts"] >= (since_ms or 0) // 3_600_000 * 3_600_000]


def test_sync_seeds_where_the_bars_start_then_continues(tmp_path, make_ticks):
    settings = _tmp_settings(tmp_path)
    universe = load_universe(settings.universe_path)
    ticks = make_ticks(hours=3)
    ohlcv = OHLCVStore(settings.data_dir / "ohlcv")
    ohlcv.upsert("dukascopy", "USD/JPY", "5m", ticks_to_ohlcv(ticks, "5m"))
    feed = _FakeFeed(ticks)

    results = sync_intrabar(settings, universe, ohlcv=ohlcv, client_for=lambda _s: feed)
    assert [(r.source, r.symbol, r.timeframe, r.ok) for r in results] == [
        ("dukascopy", "USD/JPY", PATH_LABEL, True)
    ], "only the instrument with a two-sided feed gets a path"
    first_bar = int(ohlcv.read("dukascopy", "USD/JPY", "5m")["ts"].iloc[0])
    assert feed.calls == [first_bar]

    store = IntrabarStore(settings.data_dir / "intrabar")
    last = store.last_ts("dukascopy", "USD/JPY")
    sync_intrabar(settings, universe, ohlcv=ohlcv, client_for=lambda _s: feed)
    assert feed.calls[-1] == last
    assert results[0].rows_total == len(store.read_frame("dukascopy", "USD/JPY"))
    assert feed.budgets == [BACKFILL_MAX_HOURS] * 2


def test_the_backfill_budget_finishes_three_years_in_about_a_week():
    """Sized for the cron as it actually fires here (~5 runs a day), not as written."""
    runs_per_day = 5
    days = 3 * 365 * 24 / (BACKFILL_MAX_HOURS * runs_per_day)
    assert days <= 8
    assert BACKFILL_MAX_HOURS * runs_per_day <= 24 * 7 * 48, "no more requests a day than before"


def test_sync_failure_is_reported_not_swallowed(tmp_path, make_ticks):
    settings = _tmp_settings(tmp_path)
    universe = load_universe(settings.universe_path)
    feed = _FakeFeed(make_ticks(hours=1), fail=True)
    (result,) = sync_intrabar(settings, universe, client_for=lambda _s: feed)
    assert not result.ok and "timeout" in result.error


def test_dukascopy_fetch_ticks_returns_both_sides(monkeypatch):
    import lzma
    import struct

    record = struct.Struct(">IIIff")

    def payload(_self, _url):
        raw = record.pack(5, 151_238, 151_234, 1.0, 1.0) + record.pack(9, 151_240, 151_236, 1, 1)
        compressor = lzma.LZMACompressor(format=lzma.FORMAT_ALONE)
        return compressor.compress(raw) + compressor.flush()

    monkeypatch.setattr(DukascopySource, "_get", payload)
    source = DukascopySource(max_hours=2)
    ticks = source.fetch_ticks("USD/JPY", since_ms=T0)
    assert list(ticks.columns) == ["ts", "bid", "ask"]
    assert len(ticks) == 4
    assert (ticks["ask"] > ticks["bid"]).all()
    assert ticks["ts"].is_monotonic_increasing
    assert source.last_spread["n_ticks"] == 4.0


def test_dukascopy_fetch_ticks_budget_can_be_set_per_call(monkeypatch):
    requested: list[str] = []
    monkeypatch.setattr(DukascopySource, "_get", lambda _self, url: requested.append(url) or b"")
    source = DukascopySource(max_hours=2)
    source.fetch_ticks("USD/JPY", since_ms=T0)
    assert len(requested) == 2
    source.fetch_ticks("USD/JPY", since_ms=T0, max_hours=5)
    assert len(requested) == 2 + 5, "the per-call budget overrides the instance's"
