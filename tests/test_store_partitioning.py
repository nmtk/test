"""Monthly partitioning of sub-hourly series: correctness, and the write-amplification win."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data.store import OHLCVStore, is_partitioned, month_key

MIN_5_MS = 5 * 60 * 1000


def _bars(start: str, n: int, step_ms: int = MIN_5_MS, base: float = 150.0) -> pd.DataFrame:
    ts0 = int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    ts = np.arange(n, dtype="int64") * step_ms + ts0
    close = base + np.arange(n) * 0.001
    return pd.DataFrame(
        {"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 1.0}
    )


# -- which timeframes partition ----------------------------------------------------------


def test_sub_hourly_partitions_coarse_does_not():
    assert is_partitioned("1m") and is_partitioned("5m") and is_partitioned("15m")
    assert not is_partitioned("1h")
    assert not is_partitioned("4h")
    assert not is_partitioned("1d")


def test_non_bar_timeframe_labels_are_treated_as_coarse():
    """The macro layer stores monthly releases under "1mo", which is not a bar size."""
    assert not is_partitioned("1mo")


def test_month_key_format():
    assert month_key(int(pd.Timestamp("2024-03-15", tz="UTC").value // 1_000_000)) == "2024-03"
    assert (
        month_key(int(pd.Timestamp("2024-12-31 23:59", tz="UTC").value // 1_000_000)) == "2024-12"
    )


# -- round trip --------------------------------------------------------------------------


def test_partitioned_round_trip_matches_what_was_written(tmp_path):
    store = OHLCVStore(tmp_path)
    bars = _bars("2024-01-01", 3000)  # spans several months at 5m
    store.upsert("dukascopy", "USD/JPY", "5m", bars)

    read_back = store.read("dukascopy", "USD/JPY", "5m")
    pd.testing.assert_frame_equal(read_back, bars.astype(read_back.dtypes.to_dict()))


def test_partition_files_are_one_per_month(tmp_path):
    store = OHLCVStore(tmp_path)
    # 5m bars for ~70 days -> January, February, March.
    store.upsert("dukascopy", "USD/JPY", "5m", _bars("2024-01-01", 288 * 70))
    files = sorted(p.name for p in store.partition_dir("dukascopy", "USD/JPY", "5m").glob("*"))
    assert files == ["2024-01.parquet", "2024-02.parquet", "2024-03.parquet"]


def test_coarse_timeframe_keeps_the_single_file_layout(tmp_path):
    store = OHLCVStore(tmp_path)
    store.upsert("bitbank", "BTC/JPY", "1h", _bars("2024-01-01", 500, step_ms=3_600_000))
    assert store.path("bitbank", "BTC/JPY", "1h").is_file()
    assert not store.partition_dir("bitbank", "BTC/JPY", "1h").exists()


def test_exists_reports_both_layouts(tmp_path):
    store = OHLCVStore(tmp_path)
    assert not store.exists("dukascopy", "USD/JPY", "5m")
    store.upsert("dukascopy", "USD/JPY", "5m", _bars("2024-01-01", 10))
    assert store.exists("dukascopy", "USD/JPY", "5m")


# -- the point of the exercise -----------------------------------------------------------


def test_appending_rewrites_only_the_current_month(tmp_path):
    """The whole reason partitioning exists: a sync must not rewrite years of history.

    Every earlier month's bytes must be untouched, so git stores no new blob for them.
    """
    store = OHLCVStore(tmp_path)
    store.upsert("dukascopy", "USD/JPY", "5m", _bars("2024-01-01", 288 * 70))
    directory = store.partition_dir("dukascopy", "USD/JPY", "5m")
    before = {p.name: p.read_bytes() for p in directory.glob("*.parquet")}
    assert len(before) == 3

    # One incremental sync's worth of new bars, landing in March.
    tail = store.read("dukascopy", "USD/JPY", "5m").iloc[-1:]
    new = _bars("2024-01-01", 6)
    new["ts"] = int(tail["ts"].iloc[0]) + np.arange(1, 7, dtype="int64") * MIN_5_MS
    store.upsert("dukascopy", "USD/JPY", "5m", new)

    after = {p.name: p.read_bytes() for p in directory.glob("*.parquet")}
    assert after["2024-01.parquet"] == before["2024-01.parquet"], "January must not be rewritten"
    assert after["2024-02.parquet"] == before["2024-02.parquet"], "February must not be rewritten"
    assert after["2024-03.parquet"] != before["2024-03.parquet"], "March should have grown"


def test_cross_month_append_touches_both_affected_months(tmp_path):
    store = OHLCVStore(tmp_path)
    store.upsert("dukascopy", "USD/JPY", "5m", _bars("2024-01-01", 288 * 40))  # Jan + Feb
    directory = store.partition_dir("dukascopy", "USD/JPY", "5m")
    before = {p.name: p.read_bytes() for p in directory.glob("*.parquet")}

    store.upsert("dukascopy", "USD/JPY", "5m", _bars("2024-02-25", 288 * 10))  # Feb into Mar
    after = {p.name: p.read_bytes() for p in directory.glob("*.parquet")}

    assert after["2024-01.parquet"] == before["2024-01.parquet"]
    assert after["2024-02.parquet"] != before["2024-02.parquet"]
    assert "2024-03.parquet" in after


# -- idempotency and incremental sync ----------------------------------------------------


def test_upsert_is_idempotent(tmp_path):
    store = OHLCVStore(tmp_path)
    bars = _bars("2024-01-01", 2000)
    first = store.upsert("dukascopy", "USD/JPY", "5m", bars)
    second = store.upsert("dukascopy", "USD/JPY", "5m", bars)
    assert first == second == len(bars)
    assert len(store.read("dukascopy", "USD/JPY", "5m")) == len(bars)


def test_overlapping_upsert_overwrites_rather_than_duplicating(tmp_path):
    """A re-fetched bar that was still forming must replace its stale version."""
    store = OHLCVStore(tmp_path)
    bars = _bars("2024-01-01", 100)
    store.upsert("dukascopy", "USD/JPY", "5m", bars)

    revised = bars.iloc[-5:].copy()
    revised["close"] = 999.0
    store.upsert("dukascopy", "USD/JPY", "5m", revised)

    out = store.read("dukascopy", "USD/JPY", "5m")
    assert len(out) == 100
    assert out["close"].iloc[-1] == pytest.approx(999.0)


def test_last_ts_is_correct_across_partitions(tmp_path):
    store = OHLCVStore(tmp_path)
    bars = _bars("2024-01-01", 288 * 70)
    store.upsert("dukascopy", "USD/JPY", "5m", bars)
    assert store.last_ts("dukascopy", "USD/JPY", "5m") == int(bars["ts"].iloc[-1])


def test_last_ts_on_empty_series_is_none(tmp_path):
    assert OHLCVStore(tmp_path).last_ts("dukascopy", "USD/JPY", "5m") is None


# -- migration from the pre-partitioning layout -------------------------------------------


def test_legacy_single_file_is_migrated_on_first_upsert(tmp_path):
    """A store written before partitioning must not lose data or hold it twice."""
    store = OHLCVStore(tmp_path)
    legacy_path = store.path("dukascopy", "USD/JPY", "5m")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = _bars("2024-01-01", 288 * 40)
    legacy.to_parquet(legacy_path, engine="pyarrow", index=False)

    # Readable before any migration happens.
    assert len(store.read("dukascopy", "USD/JPY", "5m")) == len(legacy)

    new = _bars("2024-01-01", 6)
    new["ts"] = int(legacy["ts"].iloc[-1]) + np.arange(1, 7, dtype="int64") * MIN_5_MS
    store.upsert("dukascopy", "USD/JPY", "5m", new)

    assert not legacy_path.exists(), "the legacy file must be removed, not left as a duplicate"
    out = store.read("dukascopy", "USD/JPY", "5m")
    assert len(out) == len(legacy) + 6
    assert out["ts"].is_monotonic_increasing
    assert not out["ts"].duplicated().any()
