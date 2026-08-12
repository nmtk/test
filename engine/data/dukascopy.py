"""Free historical FX ticks from ``datafeed.dukascopy.com``, resampled to bars.

This is the data source that makes the ``wemof`` archetype testable. The validation gate
put a number on why: on USD/JPY *daily* bars the method needs a track record of ~30,858
bars (roughly 122 years) before its Sharpe is statistically distinguishable from luck.
Three years of 5-minute bars is ~225,000 — the same three years of history, sampled ~300x
more finely, which is the regime where the minimum track-record length is satisfiable.

Dukascopy publishes one LZMA-compressed file per instrument-hour, keyless and free::

    /datafeed/{SYMBOL}/{year}/{month0}/{day}/{hour}h_ticks.bi5

``month0`` is **zero-based** (January = ``00``) — the single most common mistake when
reading this feed, and one that silently returns a different month's data rather than an
error. Each decompressed record is 20 bytes, big-endian::

    uint32  milliseconds since the start of the hour
    uint32  ask, in integer points
    uint32  bid, in integer points
    float32 ask volume
    float32 bid volume

Prices are integers scaled by the instrument's point size: JPY-quoted pairs quote to three
decimals, everything else to five.

Why ticks rather than someone else's minute bars: WEMOF targets a 1-2 pip move against a
0.2-0.5 pip spread, so the spread is 10-50% of the gross edge. A mid-price bar series
hides that entirely. These files carry both sides of every quote, so
:func:`spread_stats_bps` can measure the real cost from the same data the backtest runs
on, instead of leaving it to an assumption.
"""

from __future__ import annotations

import lzma
import struct
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from engine.core.clock import timeframe_seconds
from engine.data.schema import empty_ohlcv, normalize_ohlcv

BASE_URL = "https://datafeed.dukascopy.com/datafeed"

_RECORD = struct.Struct(">IIIff")  # ms offset, ask points, bid points, ask vol, bid vol
RECORD_SIZE = _RECORD.size  # 20 bytes

# A bounded number of hourly files per sync run. A full three-year backfill is ~26,000
# requests, which would blow a free runner's budget — and be rude to a free feed — in one
# go. The store upserts, so each scheduled run simply extends history by up to a week
# until it catches up; after that a run only walks the few hours since the last bar.
#
# This pairs with the 30-minute sync cron rather than a slower FX-specific one. At 48 runs
# a day the walk covers ~336 days of history per real day, so a three-year backfill
# completes in ~3.3 days; a daily cadence would take ~156. The size argument for a slower
# cadence is already answered by monthly partitioning in engine/data/store.py — measured
# at 0.25 MiB of git growth per simulated day, down from 12.3 MiB unpartitioned — so
# trading 47x the backfill time for it would be a bad deal.
DEFAULT_MAX_HOURS = 24 * 7

# Dukascopy has no data before this, and requesting earlier just wastes 404s.
EPOCH = datetime(2003, 5, 5, tzinfo=UTC)


def point_divisor(symbol: str) -> float:
    """Scale factor turning integer price points into real prices.

    JPY-quoted pairs are quoted to 3 decimals, every other major to 5.
    """
    return 1000.0 if symbol.upper().endswith("JPY") else 100_000.0


def to_dukascopy_symbol(symbol: str) -> str:
    """``"USD/JPY"`` -> ``"USDJPY"`` (the feed's own naming)."""
    return symbol.replace("/", "").replace("_", "").upper()


def hour_url(symbol: str, when: datetime) -> str:
    """URL of the hourly tick file covering ``when``. Note the zero-based month."""
    sym = to_dukascopy_symbol(symbol)
    return (
        f"{BASE_URL}/{sym}/{when.year:04d}/{when.month - 1:02d}/{when.day:02d}/"
        f"{when.hour:02d}h_ticks.bi5"
    )


def decode_bi5(payload: bytes, symbol: str, hour_start: datetime) -> pd.DataFrame:
    """Decode one hourly ``.bi5`` file into a tick frame.

    Returns columns ``ts`` (epoch ms), ``bid``, ``ask``. An empty payload — which the feed
    serves for hours the market was closed — decodes to an empty frame, not an error.
    """
    if not payload:
        return pd.DataFrame({"ts": [], "bid": [], "ask": []}).astype(
            {"ts": "int64", "bid": "float64", "ask": "float64"}
        )

    raw = _decompress(payload)
    usable = len(raw) - (len(raw) % RECORD_SIZE)
    if usable <= 0:
        return decode_bi5(b"", symbol, hour_start)

    divisor = point_divisor(symbol)
    hour_ms = int(hour_start.timestamp() * 1000)
    records = np.frombuffer(
        raw[:usable],
        dtype=np.dtype(
            [
                ("offset_ms", ">u4"),
                ("ask", ">u4"),
                ("bid", ">u4"),
                ("ask_vol", ">f4"),
                ("bid_vol", ">f4"),
            ]
        ),
    )
    frame = pd.DataFrame(
        {
            "ts": hour_ms + records["offset_ms"].astype("int64"),
            "bid": records["bid"].astype("float64") / divisor,
            "ask": records["ask"].astype("float64") / divisor,
        }
    )
    # A zero on either side is a malformed quote, not a real price of zero.
    return frame[(frame["bid"] > 0.0) & (frame["ask"] > 0.0)].reset_index(drop=True)


def _decompress(payload: bytes) -> bytes:
    """LZMA-decompress a bi5 body, tolerating both the alone and raw framings."""
    try:
        return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(payload)
    except lzma.LZMAError:
        # Some hours come back already decompressed (or in the auto-detected framing).
        try:
            return lzma.decompress(payload)
        except lzma.LZMAError:
            return payload


def ticks_to_ohlcv(ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a tick frame to canonical OHLCV bars built from the **mid** price.

    Mid is the honest reference for a backtest: it is neither side's tradeable price, so
    costs stay explicit in the fee/slippage model rather than being smuggled into the bar.
    ``volume`` is the tick count, which is the only volume FX actually has.
    """
    if len(ticks) == 0:
        return empty_ohlcv()
    step_ms = timeframe_seconds(timeframe) * 1000
    mid = (ticks["bid"].to_numpy() + ticks["ask"].to_numpy()) / 2.0
    bucket = (ticks["ts"].to_numpy(dtype="int64") // step_ms) * step_ms
    frame = pd.DataFrame({"bucket": bucket, "mid": mid})
    grouped = frame.groupby("bucket", sort=True)["mid"]
    bars = pd.DataFrame(
        {
            "ts": grouped.first().index.to_numpy(dtype="int64"),
            "open": grouped.first().to_numpy(),
            "high": grouped.max().to_numpy(),
            "low": grouped.min().to_numpy(),
            "close": grouped.last().to_numpy(),
            "volume": grouped.count().to_numpy(dtype="float64"),
        }
    )
    return normalize_ohlcv(bars)


def spread_stats_bps(ticks: pd.DataFrame) -> dict[str, float]:
    """Measured bid/ask spread in basis points of mid — the real cost of trading this.

    Feed these numbers to ``--slippage-bps`` instead of guessing. For a scalp whose target
    is a couple of pips, the difference between an assumed and a measured spread decides
    whether the strategy is profitable at all.
    """
    if len(ticks) == 0:
        return {"median_bps": 0.0, "mean_bps": 0.0, "p90_bps": 0.0, "n_ticks": 0.0}
    bid = ticks["bid"].to_numpy()
    ask = ticks["ask"].to_numpy()
    mid = (bid + ask) / 2.0
    bps = np.where(mid > 0.0, (ask - bid) / mid * 10_000.0, np.nan)
    bps = bps[np.isfinite(bps)]
    if len(bps) == 0:
        return {"median_bps": 0.0, "mean_bps": 0.0, "p90_bps": 0.0, "n_ticks": 0.0}
    return {
        "median_bps": float(np.median(bps)),
        "mean_bps": float(np.mean(bps)),
        "p90_bps": float(np.percentile(bps, 90)),
        "n_ticks": float(len(bps)),
    }


class FeedUnreachable(RuntimeError):
    """The feed host could not be reached at all, as opposed to an hour having no data.

    Worth its own type because the two failures call for opposite responses: a missing
    hour is normal and skipped, while an unreachable host means this source cannot serve
    the instrument from this network and the operator needs to switch to another one.
    """


@dataclass(slots=True)
class DukascopySource:
    """Historical FX bars built from Dukascopy ticks.

    ``symbol`` is the canonical engine symbol (``"USD/JPY"``); it is translated to the
    feed's naming internally. Each ``fetch_ohlcv`` call walks at most ``max_hours`` hourly
    files, so a long backfill converges over several scheduled runs rather than hanging a
    single one.

    Reachability caveat: this feed is not guaranteed to answer datacenter IPs. This repo
    already carries one instance of that problem — stooq 404s every GitHub Actions runner
    (see strategies_config/context.yaml) — and Dukascopy has been observed *timing out at
    the TCP level* from Actions, which no header or retry can fix. When that happens the
    fix is not here: point the instrument's ``data.source`` at ``oanda`` in
    universe.yaml, which serves the same 5m candles over an API built for programmatic
    access (free practice credentials, see engine/execution/brokers/oanda.py).
    """

    max_hours: int = DEFAULT_MAX_HOURS
    timeout: int = 30
    retries: int = 3
    # Populated after each fetch so callers can report the measured spread.
    last_spread: dict[str, float] | None = None
    _session: object | None = None

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        start = (
            datetime.fromtimestamp(since_ms / 1000, tz=UTC) if since_ms is not None else None
        ) or EPOCH
        start = max(start.replace(minute=0, second=0, microsecond=0), EPOCH)
        # The current hour is still forming; stop at the last hour that has fully closed.
        end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

        frames: list[pd.DataFrame] = []
        cursor = start
        fetched = 0
        attempted = 0
        failures = 0
        while cursor < end and fetched < self.max_hours:
            if cursor.weekday() < 5 or cursor.weekday() == 6:
                # Saturday has no session; Sunday only opens late, but the feed simply
                # returns an empty body for closed hours so it costs one cheap request.
                attempted += 1
                try:
                    payload = self._get(hour_url(symbol, cursor))
                except Exception:  # noqa: BLE001 - one bad hour must not lose the run
                    failures += 1
                    # Every one of the first few requests failing means the host is not
                    # answering this network at all; anything else is a flaky hour to skip.
                    if failures == attempted and attempted >= 3:
                        raise FeedUnreachable(
                            f"{BASE_URL} did not answer {attempted} consecutive requests — "
                            "the feed is likely blocking this network. Point this "
                            "instrument's data.source at 'oanda' in universe.yaml."
                        ) from None
                else:
                    if payload:
                        frames.append(decode_bi5(payload, symbol, cursor))
            cursor += timedelta(hours=1)
            fetched += 1

        if not frames:
            self.last_spread = spread_stats_bps(pd.DataFrame({"bid": [], "ask": []}))
            return empty_ohlcv()

        ticks = pd.concat(frames, ignore_index=True).sort_values("ts", kind="mergesort")
        self.last_spread = spread_stats_bps(ticks)
        bars = ticks_to_ohlcv(ticks, timeframe)
        if limit is not None and len(bars) > limit:
            bars = bars.tail(limit).reset_index(drop=True)
        return bars

    def _client(self):
        """A pooled session — a single run makes up to ``max_hours`` sequential requests."""
        if self._session is None:
            import requests

            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
                    ),
                    "Accept": "*/*",
                }
            )
            self._session = session
        return self._session

    def _get(self, url: str) -> bytes:
        """Fetch one hourly file. Returns empty bytes for an hour with no session."""
        import time

        last_error: Exception | None = None
        for attempt in range(max(self.retries, 1)):
            try:
                response = self._client().get(url, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised if terminal
                last_error = exc
                time.sleep(2**attempt)
                continue
            if response.status_code == 404:
                return b""  # no session in that hour
            if response.status_code >= 500:
                last_error = RuntimeError(f"{response.status_code} from {url}")
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.content
        raise last_error if last_error else RuntimeError(f"failed to fetch {url}")
