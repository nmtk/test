"""Bar data sources.

Each source implements :class:`BarSource.fetch_ohlcv` and returns a canonical OHLCV
frame (see :mod:`engine.data.schema`). Heavy third-party libraries are imported lazily so
this module imports cleanly even when a given backend isn't installed.

* :class:`CcxtSource` — ccxt REST (``bitbank`` live venue, ``binance`` research top-ups).
* :class:`AlpacaSource` — alpaca-py historical bars (free IEX feed by default).
* :class:`BinanceVisionSource` — bulk monthly klines from ``data.binance.vision`` for
  deep backfill (1m history back to ~2017).
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from typing import Protocol

import pandas as pd

from engine.core.clock import timeframe_seconds
from engine.data.schema import OHLCV_COLUMNS, empty_ohlcv, normalize_ohlcv

# A defensive cap so a misconfigured pagination loop can never run away on a free runner.
_MAX_PAGES = 5000


class BarSource(Protocol):
    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Return canonical OHLCV bars at/after ``since_ms`` (epoch ms), ascending."""
        ...


class CcxtSource:
    """OHLCV via ccxt. Used for bitbank (live venue) and binance (research top-ups)."""

    def __init__(self, exchange_id: str, api_key: str | None = None, api_secret: str | None = None):
        self.exchange_id = exchange_id
        self._api_key = api_key
        self._api_secret = api_secret
        self._exchange = None

    def _client(self):
        if self._exchange is None:
            import ccxt

            klass = getattr(ccxt, self.exchange_id)
            config: dict = {"enableRateLimit": True}
            if self._api_key and self._api_secret:
                config["apiKey"] = self._api_key
                config["secret"] = self._api_secret
            self._exchange = klass(config)
        return self._exchange

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        client = self._client()
        step_ms = timeframe_seconds(timeframe) * 1000
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        cursor = since_ms
        rows: list[list] = []
        pages = 0
        while pages < _MAX_PAGES:
            batch = client.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1][0]
            next_cursor = last_ts + step_ms
            # Stop when we stop making progress or have caught up to ~now.
            if next_cursor <= (cursor or 0) or last_ts >= now_ms - step_ms:
                break
            cursor = next_cursor
            pages += 1
            if limit is not None and len(batch) < limit:
                break
        if not rows:
            return empty_ohlcv()
        df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
        return normalize_ohlcv(df)


class AlpacaSource:
    """OHLCV via alpaca-py historical data (defaults to the free IEX feed)."""

    def __init__(
        self, api_key: str | None = None, api_secret: str | None = None, feed: str = "iex"
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self.feed = feed
        self._client = None

    def _data_client(self):
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(self._api_key, self._api_secret)
        return self._client

    def _timeframe(self, timeframe: str):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        qty = int(timeframe[:-1])
        unit = timeframe[-1]
        mapping = {
            "m": TimeFrameUnit.Minute,
            "h": TimeFrameUnit.Hour,
            "d": TimeFrameUnit.Day,
            "w": TimeFrameUnit.Week,
        }
        if unit not in mapping:
            raise ValueError(f"unsupported alpaca timeframe: {timeframe!r}")
        return TimeFrame(qty, mapping[unit])

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest

        start = datetime.fromtimestamp(since_ms / 1000, tz=UTC) if since_ms is not None else None
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._timeframe(timeframe),
            start=start,
            limit=limit,
            feed=DataFeed(self.feed),
        )
        barset = self._data_client().get_stock_bars(request)
        bars = barset.data.get(symbol, [])
        if not bars:
            return empty_ohlcv()
        rows = [
            [int(b.timestamp.timestamp() * 1000), b.open, b.high, b.low, b.close, b.volume]
            for b in bars
        ]
        df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
        return normalize_ohlcv(df)


class BinanceVisionSource:
    """Bulk monthly klines from ``data.binance.vision`` for deep research backfill.

    ``symbol`` here is the Binance market symbol without a slash, e.g. ``"BTCUSDT"``.
    """

    BASE = "https://data.binance.vision/data/spot/monthly/klines"

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        import requests

        start = (
            datetime.fromtimestamp(since_ms / 1000, tz=UTC)
            if since_ms is not None
            else datetime(2017, 1, 1, tzinfo=UTC)
        )
        now = datetime.now(UTC)
        frames: list[pd.DataFrame] = []
        year, month = start.year, start.month
        while (year, month) <= (now.year, now.month):
            url = (
                f"{self.BASE}/{symbol}/{timeframe}/{symbol}-{timeframe}-{year:04d}-{month:02d}.zip"
            )
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                frames.append(self._parse_zip(resp.content))
            month += 1
            if month > 12:
                month = 1
                year += 1
        if not frames:
            return empty_ohlcv()
        df = pd.concat(frames, ignore_index=True)
        if since_ms is not None:
            df = df[df["ts"] >= since_ms]
        return normalize_ohlcv(df)

    @staticmethod
    def _parse_zip(content: bytes) -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                raw = pd.read_csv(fh, header=None)
        # Binance klines CSV: open_time, open, high, low, close, volume, close_time, ...
        df = raw.iloc[:, :6].copy()
        df.columns = OHLCV_COLUMNS
        # Newer dumps use microsecond open_time; normalize to milliseconds.
        if df["ts"].iloc[0] > 1e15:
            df["ts"] = (df["ts"] // 1000).astype("int64")
        return df


class StooqSource:
    """Daily bars from stooq.com's free CSV endpoint (no API key).

    Used for macro *context* series (equity indices, gold, yields, FX) — research inputs,
    never traded. ``symbol`` is stooq-native, e.g. ``^spx``, ``xauusd``, ``10yusy.b``.
    """

    BASE = "https://stooq.com/q/d/l/"
    # stooq answers 404 to non-browser agents (the default python-requests UA gets
    # rejected on datacenter IPs like Actions runners), so present a browser UA.
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        import requests

        if timeframe != "1d":
            raise ValueError(f"stooq source only serves 1d bars, got {timeframe!r}")
        resp = requests.get(
            self.BASE, params={"s": symbol, "i": "d"}, headers=self.HEADERS, timeout=60
        )
        resp.raise_for_status()
        df = self._parse_csv(resp.text)
        if since_ms is not None:
            df = df[df["ts"] >= since_ms]
        return normalize_ohlcv(df) if len(df) else empty_ohlcv()

    @staticmethod
    def _parse_csv(text: str) -> pd.DataFrame:
        """Parse stooq CSV (``Date,Open,High,Low,Close[,Volume]``) to canonical OHLCV."""
        raw = pd.read_csv(io.StringIO(text))
        raw.columns = [c.strip().lower() for c in raw.columns]
        if "date" not in raw.columns or "close" not in raw.columns:
            return empty_ohlcv()  # e.g. "No data" body for an unknown symbol
        out = pd.DataFrame()
        out["ts"] = (
            pd.to_datetime(raw["date"], utc=True).astype("datetime64[ms, UTC]").astype("int64")
        )
        for col in ("open", "high", "low", "close"):
            source_col = col if col in raw.columns else "close"
            out[col] = pd.to_numeric(raw[source_col], errors="coerce")
        if "volume" in raw.columns:
            out["volume"] = pd.to_numeric(raw["volume"], errors="coerce").fillna(0.0)
        else:
            out["volume"] = 0.0
        return out.dropna(subset=["close"])


class FredSource:
    """Observations from FRED's keyless ``fredgraph.csv`` endpoint.

    Used for context series (e.g. ``DGS10``) and for released statistics (the monthly
    trade-flow series). Values are levels, so open=high=low=close and volume=0.

    ``timeframe`` here is a **storage label, not a request parameter**: fredgraph.csv
    always returns a series at its own native frequency, so the caller's job is to label
    it correctly, not to ask for a resampling FRED will not do. Daily series are stored
    under ``1d`` and monthly ones under ``1mo``; anything else is a configuration mistake
    worth failing on.
    """

    BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    SUPPORTED_TIMEFRAMES = ("1d", "1mo")

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        import requests

        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"fred source stores {self.SUPPORTED_TIMEFRAMES} series, got {timeframe!r}"
            )
        resp = requests.get(self.BASE, params={"id": symbol}, timeout=60)
        resp.raise_for_status()
        df = self._parse_csv(resp.text, symbol)
        if since_ms is not None:
            df = df[df["ts"] >= since_ms]
        return normalize_ohlcv(df) if len(df) else empty_ohlcv()

    @staticmethod
    def _parse_csv(text: str, symbol: str) -> pd.DataFrame:
        raw = pd.read_csv(io.StringIO(text))
        raw.columns = [c.strip() for c in raw.columns]
        date_col = next((c for c in raw.columns if c.upper() in ("DATE", "OBSERVATION_DATE")), None)
        value_col = next((c for c in raw.columns if c != date_col), None)
        if date_col is None or value_col is None:
            return empty_ohlcv()
        value = pd.to_numeric(raw[value_col], errors="coerce")  # missing prints are "."
        out = pd.DataFrame()
        out["ts"] = (
            pd.to_datetime(raw[date_col], utc=True).astype("datetime64[ms, UTC]").astype("int64")
        )
        for col in ("open", "high", "low", "close"):
            out[col] = value
        out["volume"] = 0.0
        return out.dropna(subset=["close"])


# Keyless sources that are also expected to be *reliable*. A failure from one of these is
# a genuine breakage — a dead endpoint, a blocked network, a wrong series id — rather than
# the expected no-op of running without a venue's credentials configured, so it turns the
# sync job red (see ``_cmd_sync_data``).
#
# stooq is deliberately absent: it is keyless but documented as best-effort here because
# it 404s datacenter IPs, and the feature layer already degrades its series to neutral.
# Holding it to this bar would leave the job permanently red for a known, accepted gap.
KEYLESS_SOURCES: frozenset[str] = frozenset({"fred", "dukascopy", "binance", "bitbank"})


def make_source(source: str, settings) -> BarSource:
    """Factory: map a universe source name to a concrete :class:`BarSource`."""
    if source == "stooq":
        return StooqSource()
    if source == "dukascopy":
        from engine.data.dukascopy import DukascopySource

        return DukascopySource()
    if source == "oanda":
        from engine.execution.brokers.oanda import OandaBroker

        return OandaBroker.from_settings(settings)
    if source == "fred":
        return FredSource()
    if source == "bitbank":
        return CcxtSource(
            "bitbank",
            api_key=settings.bitbank.api_key,
            api_secret=settings.bitbank.api_secret,
        )
    if source == "binance":
        return CcxtSource("binance")
    if source == "alpaca":
        return AlpacaSource(
            api_key=settings.alpaca.api_key,
            api_secret=settings.alpaca.api_secret,
        )
    raise ValueError(f"unknown data source: {source!r}")
