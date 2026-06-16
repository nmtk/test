import pandas as pd

from engine.data.schema import OHLCV_COLUMNS, merge_ohlcv, normalize_ohlcv, with_datetime
from engine.data.store import OHLCVStore, sanitize_symbol


def _df(rows):
    return pd.DataFrame(rows, columns=OHLCV_COLUMNS)


def test_normalize_sorts_and_dedupes():
    df = _df(
        [
            [2000, 2, 2, 2, 2, 2],
            [1000, 1, 1, 1, 1, 1],
            [2000, 9, 9, 9, 9, 9],  # duplicate ts; last wins
        ]
    )
    out = normalize_ohlcv(df)
    assert list(out["ts"]) == [1000, 2000]
    assert out["close"].iloc[-1] == 9.0
    assert out["ts"].dtype == "int64"


def test_merge_is_idempotent():
    a = _df([[1000, 1, 1, 1, 1, 1], [2000, 2, 2, 2, 2, 2]])
    b = _df([[2000, 2, 2, 2, 2, 2], [3000, 3, 3, 3, 3, 3]])
    merged = merge_ohlcv(a, b)
    assert list(merged["ts"]) == [1000, 2000, 3000]
    # merging again changes nothing
    assert merge_ohlcv(merged, b).equals(merged)


def test_merge_incoming_overwrites():
    a = _df([[1000, 1, 1, 1, 1, 1]])
    b = _df([[1000, 5, 5, 5, 5, 5]])
    merged = merge_ohlcv(a, b)
    assert merged["close"].iloc[0] == 5.0


def test_with_datetime():
    out = with_datetime(_df([[0, 1, 1, 1, 1, 1]]))
    assert str(out["dt"].iloc[0]) == "1970-01-01 00:00:00+00:00"


def test_sanitize_symbol():
    assert sanitize_symbol("BTC/JPY") == "BTC_JPY"
    assert sanitize_symbol("BTC/USDT:USDT") == "BTC_USDT_USDT"


def test_store_roundtrip(tmp_path):
    store = OHLCVStore(tmp_path)
    assert store.last_ts("bitbank", "BTC/JPY", "1h") is None

    store.upsert("bitbank", "BTC/JPY", "1h", _df([[1000, 1, 1, 1, 1, 1], [2000, 2, 2, 2, 2, 2]]))
    assert store.last_ts("bitbank", "BTC/JPY", "1h") == 2000

    # incremental upsert merges, re-fetching the last bar is harmless
    n = store.upsert(
        "bitbank", "BTC/JPY", "1h", _df([[2000, 2, 2, 2, 2, 2], [3000, 3, 3, 3, 3, 3]])
    )
    assert n == 3
    back = store.read("bitbank", "BTC/JPY", "1h")
    assert list(back["ts"]) == [1000, 2000, 3000]


def test_store_read_range(tmp_path):
    store = OHLCVStore(tmp_path)
    store.upsert("binance", "BTC/USDT", "1d", _df([[i * 1000, i, i, i, i, i] for i in range(1, 6)]))
    bounded = store.read("binance", "BTC/USDT", "1d", start_ms=2000, end_ms=4000)
    assert list(bounded["ts"]) == [2000, 3000]
