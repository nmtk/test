import pandas as pd
import pytest

from engine.data.resample import resample_ohlcv
from engine.data.schema import OHLCV_COLUMNS, normalize_ohlcv
from engine.data.store import OHLCVStore
from engine.data.sync import derive_series

H = 3_600_000  # one hour in ms


def _hourly(n, start=0):
    rows = [[start + i * H, i + 1, i + 1, i + 1, i + 1, 1.0] for i in range(n)]
    return pd.DataFrame(rows, columns=OHLCV_COLUMNS)


def test_resample_1h_to_4h_ohlcv():
    out = resample_ohlcv(_hourly(8), "1h", "4h")
    assert list(out["ts"]) == [0, 4 * H]
    assert list(out["open"]) == [1, 5]
    assert list(out["high"]) == [4, 8]
    assert list(out["low"]) == [1, 5]
    assert list(out["close"]) == [4, 8]
    assert list(out["volume"]) == [4, 4]


def test_incomplete_bucket_dropped():
    # 6 hourly bars => one complete 4h bucket (first 4); the trailing 2 are dropped
    out = resample_ohlcv(_hourly(6), "1h", "4h")
    assert list(out["ts"]) == [0]


def test_ratio_one_is_identity():
    df = _hourly(5)
    assert resample_ohlcv(df, "1h", "1h").equals(normalize_ohlcv(df))


def test_non_multiple_raises():
    with pytest.raises(ValueError):
        resample_ohlcv(_hourly(10), "4h", "6h")  # 6h is not a multiple of 4h


def test_derive_series_into_store(tmp_path):
    store = OHLCVStore(tmp_path)
    store.upsert("bitbank", "BTC/JPY", "1h", _hourly(48))  # two full days
    r4 = derive_series(store, "bitbank", "BTC/JPY", "1h", "4h")
    r1d = derive_series(store, "bitbank", "BTC/JPY", "1h", "1d")
    assert r4.ok and r4.rows_total == 12  # 48h / 4h
    assert r1d.ok and r1d.rows_total == 2  # 48h / 24h
    assert store.read("bitbank", "BTC/JPY", "1d")["close"].iloc[0] == 24


def test_derive_series_no_base(tmp_path):
    store = OHLCVStore(tmp_path)
    r = derive_series(store, "bitbank", "BTC/JPY", "1h", "4h")
    assert not r.ok
    assert r.error == "no base data"
