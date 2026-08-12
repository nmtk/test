"""Dukascopy tick-feed tests.

The network is not reachable from CI's sandbox and must never be a test dependency
anyway, so every test here builds real ``.bi5`` payloads with the same binary layout the
feed serves and drives the decoder end to end.
"""

from __future__ import annotations

import lzma
import struct
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from engine.data.dukascopy import (
    RECORD_SIZE,
    DukascopySource,
    decode_bi5,
    hour_url,
    point_divisor,
    spread_stats_bps,
    ticks_to_ohlcv,
    to_dukascopy_symbol,
)

RECORD = struct.Struct(">IIIff")


def _bi5(records: list[tuple[int, int, int, float, float]]) -> bytes:
    """Build a compressed hourly payload exactly as the feed serves it."""
    raw = b"".join(RECORD.pack(*r) for r in records)
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_ALONE)
    return compressor.compress(raw) + compressor.flush()


# -- URL construction --------------------------------------------------------------------


def test_month_in_the_url_is_zero_based():
    """January is 00. Getting this wrong silently serves a different month."""
    assert hour_url("USD/JPY", datetime(2024, 1, 15, 9, tzinfo=UTC)).endswith(
        "/USDJPY/2024/00/15/09h_ticks.bi5"
    )
    assert hour_url("USD/JPY", datetime(2024, 12, 3, 0, tzinfo=UTC)).endswith(
        "/USDJPY/2024/11/03/00h_ticks.bi5"
    )


def test_symbol_translation():
    assert to_dukascopy_symbol("USD/JPY") == "USDJPY"
    assert to_dukascopy_symbol("EUR/USD") == "EURUSD"
    assert to_dukascopy_symbol("usdjpy") == "USDJPY"


def test_point_divisor_matches_quote_convention():
    assert point_divisor("USD/JPY") == 1000.0  # 3 decimals
    assert point_divisor("EUR/JPY") == 1000.0
    assert point_divisor("EUR/USD") == 100_000.0  # 5 decimals


# -- decoding ----------------------------------------------------------------------------


def test_decode_round_trip_recovers_prices_and_timestamps():
    hour = datetime(2024, 3, 5, 13, tzinfo=UTC)
    hour_ms = int(hour.timestamp() * 1000)
    # USD/JPY at 151.234 / 151.238 -> integer points at 3 decimals.
    records = [
        (0, 151_238, 151_234, 1.5, 2.0),
        (1_500, 151_240, 151_236, 1.0, 1.0),
        (3_599_999, 151_250, 151_245, 0.5, 0.5),
    ]
    ticks = decode_bi5(_bi5(records), "USD/JPY", hour)

    assert len(ticks) == 3
    assert list(ticks["ts"]) == [hour_ms, hour_ms + 1_500, hour_ms + 3_599_999]
    assert ticks["bid"].iloc[0] == pytest.approx(151.234)
    assert ticks["ask"].iloc[0] == pytest.approx(151.238)
    assert ticks["bid"].iloc[2] == pytest.approx(151.245)
    assert (ticks["ask"] > ticks["bid"]).all()


def test_decode_uses_five_decimals_for_non_jpy_pairs():
    hour = datetime(2024, 3, 5, 13, tzinfo=UTC)
    ticks = decode_bi5(_bi5([(0, 108_765, 108_760, 1.0, 1.0)]), "EUR/USD", hour)
    assert ticks["bid"].iloc[0] == pytest.approx(1.08760)
    assert ticks["ask"].iloc[0] == pytest.approx(1.08765)


def test_empty_payload_is_an_empty_frame_not_an_error():
    """Closed hours (weekends, holidays) legitimately serve nothing."""
    ticks = decode_bi5(b"", "USD/JPY", datetime(2024, 3, 9, 3, tzinfo=UTC))
    assert len(ticks) == 0
    assert list(ticks.columns) == ["ts", "bid", "ask"]
    assert ticks["ts"].dtype == np.dtype("int64")


def test_truncated_trailing_record_is_ignored():
    hour = datetime(2024, 3, 5, 13, tzinfo=UTC)
    raw = RECORD.pack(0, 151_238, 151_234, 1.0, 1.0) + b"\x00\x01\x02"  # half a record
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_ALONE)
    payload = compressor.compress(raw) + compressor.flush()
    ticks = decode_bi5(payload, "USD/JPY", hour)
    assert len(ticks) == 1, "a partial trailing record must be dropped, not misread"


def test_zero_prices_are_dropped_as_malformed():
    hour = datetime(2024, 3, 5, 13, tzinfo=UTC)
    records = [(0, 151_238, 151_234, 1.0, 1.0), (10, 0, 0, 1.0, 1.0)]
    ticks = decode_bi5(_bi5(records), "USD/JPY", hour)
    assert len(ticks) == 1


def test_record_size_is_twenty_bytes():
    assert RECORD_SIZE == 20


# -- resampling --------------------------------------------------------------------------


def test_ticks_resample_to_correct_ohlc_buckets():
    step_ms = 5 * 60 * 1000
    base = 1_700_000_000_000 // step_ms * step_ms  # align to a bucket boundary
    # Two 5m buckets, with a known high and low in the first.
    ticks = pd.DataFrame(
        {
            "ts": [base, base + 1000, base + 2000, base + step_ms, base + step_ms + 10],
            "bid": [150.00, 150.10, 149.90, 151.00, 151.20],
            "ask": [150.00, 150.10, 149.90, 151.00, 151.20],
        }
    )
    bars = ticks_to_ohlcv(ticks, "5m")
    assert len(bars) == 2
    assert bars["ts"].iloc[0] == base
    assert bars["open"].iloc[0] == pytest.approx(150.00)
    assert bars["high"].iloc[0] == pytest.approx(150.10)
    assert bars["low"].iloc[0] == pytest.approx(149.90)
    assert bars["close"].iloc[0] == pytest.approx(149.90)
    assert bars["volume"].iloc[0] == 3.0  # tick count
    assert bars["ts"].iloc[1] == base + step_ms
    assert bars["close"].iloc[1] == pytest.approx(151.20)


def test_bars_are_built_from_the_mid_not_one_side():
    base = 1_700_000_000_000 // (5 * 60 * 1000) * (5 * 60 * 1000)
    ticks = pd.DataFrame({"ts": [base], "bid": [150.00], "ask": [150.02]})
    bars = ticks_to_ohlcv(ticks, "5m")
    assert bars["close"].iloc[0] == pytest.approx(150.01)


def test_empty_ticks_resample_to_empty_bars():
    empty = pd.DataFrame({"ts": [], "bid": [], "ask": []})
    assert len(ticks_to_ohlcv(empty, "5m")) == 0


# -- spread measurement ------------------------------------------------------------------


def test_spread_stats_measure_the_real_cost():
    """0.4 pips on USD/JPY at ~150 is ~0.027%, i.e. ~0.27bps."""
    ticks = pd.DataFrame({"ts": [0, 1, 2], "bid": [150.000] * 3, "ask": [150.004] * 3})
    stats = spread_stats_bps(ticks)
    assert stats["median_bps"] == pytest.approx(0.004 / 150.002 * 10_000, rel=1e-6)
    assert stats["median_bps"] < 0.3
    assert stats["n_ticks"] == 3.0


def test_spread_stats_on_no_ticks_are_zero_not_nan():
    stats = spread_stats_bps(pd.DataFrame({"bid": [], "ask": []}))
    assert stats == {"median_bps": 0.0, "mean_bps": 0.0, "p90_bps": 0.0, "n_ticks": 0.0}


# -- the source, with the network stubbed ------------------------------------------------


def test_fetch_walks_hours_and_stops_at_max_hours(monkeypatch):
    hour = datetime(2024, 3, 5, 0, tzinfo=UTC)
    requested: list[str] = []

    def fake_get(self, url):
        requested.append(url)
        return _bi5([(0, 151_238, 151_234, 1.0, 1.0)])

    monkeypatch.setattr(DukascopySource, "_get", fake_get)
    source = DukascopySource(max_hours=3)
    bars = source.fetch_ohlcv("USD/JPY", "1h", since_ms=int(hour.timestamp() * 1000))

    assert len(requested) == 3, "max_hours must bound a single run's request budget"
    assert len(bars) == 3
    assert source.last_spread is not None and source.last_spread["n_ticks"] == 3.0


def test_fetch_skips_saturday(monkeypatch):
    saturday = datetime(2024, 3, 9, 0, tzinfo=UTC)
    assert saturday.weekday() == 5
    requested: list[str] = []

    def fake_get(self, url):
        requested.append(url)
        return b""

    monkeypatch.setattr(DukascopySource, "_get", fake_get)
    DukascopySource(max_hours=5).fetch_ohlcv(
        "USD/JPY", "5m", since_ms=int(saturday.timestamp() * 1000)
    )
    assert requested == [], "Saturday has no FX session — do not spend requests on it"


def test_fetch_with_no_data_returns_empty_canonical_frame(monkeypatch):
    monkeypatch.setattr(DukascopySource, "_get", lambda self, url: b"")
    bars = DukascopySource(max_hours=2).fetch_ohlcv(
        "USD/JPY", "5m", since_ms=int(datetime(2024, 3, 5, tzinfo=UTC).timestamp() * 1000)
    )
    assert len(bars) == 0
    assert list(bars.columns) == ["ts", "open", "high", "low", "close", "volume"]


# -- reachability ------------------------------------------------------------------------


def test_total_failure_raises_feed_unreachable_not_silence(monkeypatch):
    """A host that answers nothing must be loud. Silence here cost a whole sync run.

    The first deployment failed exactly this way: every request timed out, the per-series
    exception handler swallowed it, and the job reported success with no FX data.
    """
    from engine.data.dukascopy import FeedUnreachable

    def always_fails(self, url):
        raise OSError("connect timeout")

    monkeypatch.setattr(DukascopySource, "_get", always_fails)
    with pytest.raises(FeedUnreachable, match="oanda"):
        DukascopySource(max_hours=10).fetch_ohlcv(
            "USD/JPY", "5m", since_ms=int(datetime(2024, 3, 5, tzinfo=UTC).timestamp() * 1000)
        )


def test_a_few_flaky_hours_do_not_lose_the_run(monkeypatch):
    """One bad hour is skipped; the surrounding good hours still make it into the store."""
    calls = {"n": 0}

    def flaky(self, url):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("transient")
        return _bi5([(0, 151_238, 151_234, 1.0, 1.0)])

    monkeypatch.setattr(DukascopySource, "_get", flaky)
    bars = DukascopySource(max_hours=5).fetch_ohlcv(
        "USD/JPY", "1h", since_ms=int(datetime(2024, 3, 5, tzinfo=UTC).timestamp() * 1000)
    )
    assert len(bars) == 4, "only the flaky hour should be missing"


def test_retries_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    class _Resp:
        status_code = 200
        content = _bi5([(0, 151_238, 151_234, 1.0, 1.0)])

        def raise_for_status(self):
            pass

    class _Session:
        headers: dict = {}

        def get(self, url, timeout=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise OSError("flaky")
            return _Resp()

    monkeypatch.setattr("time.sleep", lambda _s: None)
    source = DukascopySource(retries=3)
    source._session = _Session()
    assert source._get("http://example/x.bi5")
    assert attempts["n"] == 3
