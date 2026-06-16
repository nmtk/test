"""Timeframe math and bar-boundary helpers.

The executor is intentionally **low-latency-independent**: it derives the "current bar"
from the timestamp of the latest stored data, not from the wall clock. These helpers
make that derivation explicit and testable, and they are the reason ~20–30 min of cron
jitter is harmless when trading coarse (1h/4h/1d) closed bars.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

_UNIT_SECONDS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def parse_timeframe(timeframe: str) -> timedelta:
    """Parse a timeframe like ``"1m"``, ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``, ``"1w"``."""
    tf = timeframe.strip().lower()
    if len(tf) < 2 or not tf[:-1].isdigit():
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    qty = int(tf[:-1])
    unit = tf[-1]
    if qty <= 0 or unit not in _UNIT_SECONDS:
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    return timedelta(seconds=qty * _UNIT_SECONDS[unit])


def timeframe_seconds(timeframe: str) -> int:
    return int(parse_timeframe(timeframe).total_seconds())


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def floor_to_timeframe(ts: datetime, timeframe: str) -> datetime:
    """Floor ``ts`` (UTC) down to the start of its timeframe bucket.

    Buckets are anchored at the Unix epoch, matching exchange bar boundaries for
    sub-weekly timeframes. (Weekly anchoring at the epoch lands on Thursdays; weekly
    bars are out of scope for the executor and only used in research.)
    """
    ts = _as_utc(ts)
    step = timeframe_seconds(timeframe)
    epoch_seconds = int(ts.timestamp())
    floored = epoch_seconds - (epoch_seconds % step)
    return datetime.fromtimestamp(floored, tz=UTC)


def last_closed_bar_open(now: datetime, timeframe: str) -> datetime:
    """Return the open timestamp of the most recently *closed* bar.

    The bar opening at ``floor(now)`` is still forming (it closes at ``floor(now)+tf``),
    so the last closed bar opened one timeframe earlier.
    """
    return floor_to_timeframe(now, timeframe) - parse_timeframe(timeframe)


def is_bar_closed(bar_open: datetime, timeframe: str, now: datetime | None = None) -> bool:
    """True if the bar that opened at ``bar_open`` has fully closed by ``now``."""
    now = _as_utc(now) if now is not None else datetime.now(UTC)
    return now >= _as_utc(bar_open) + parse_timeframe(timeframe)


def utc_now() -> datetime:
    return datetime.now(UTC)
