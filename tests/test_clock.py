from datetime import UTC, datetime, timedelta

import pytest

from engine.core.clock import (
    floor_to_timeframe,
    is_bar_closed,
    last_closed_bar_open,
    parse_timeframe,
    timeframe_seconds,
)


def test_parse_timeframe():
    assert parse_timeframe("1m") == timedelta(minutes=1)
    assert parse_timeframe("15m") == timedelta(minutes=15)
    assert parse_timeframe("1h") == timedelta(hours=1)
    assert parse_timeframe("4h") == timedelta(hours=4)
    assert parse_timeframe("1d") == timedelta(days=1)
    assert parse_timeframe("1w") == timedelta(weeks=1)
    assert timeframe_seconds("1h") == 3600


@pytest.mark.parametrize("bad", ["", "h", "0h", "-1h", "1x", "1.5h", "abc"])
def test_parse_timeframe_invalid(bad):
    with pytest.raises(ValueError):
        parse_timeframe(bad)


def test_floor_to_timeframe():
    ts = datetime(2025, 1, 1, 12, 34, 56, tzinfo=UTC)
    assert floor_to_timeframe(ts, "1h") == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    assert floor_to_timeframe(ts, "1d") == datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    ts2 = datetime(2025, 1, 1, 13, 34, tzinfo=UTC)
    assert floor_to_timeframe(ts2, "4h") == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def test_floor_assumes_utc_for_naive():
    naive = datetime(2025, 1, 1, 12, 30)
    assert floor_to_timeframe(naive, "1h") == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def test_last_closed_bar_open():
    on_boundary = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    assert last_closed_bar_open(on_boundary, "1h") == datetime(2025, 1, 1, 11, 0, tzinfo=UTC)
    mid = datetime(2025, 1, 1, 12, 30, tzinfo=UTC)
    assert last_closed_bar_open(mid, "1h") == datetime(2025, 1, 1, 11, 0, tzinfo=UTC)


def test_is_bar_closed():
    bar_open = datetime(2025, 1, 1, 11, 0, tzinfo=UTC)
    assert is_bar_closed(bar_open, "1h", now=datetime(2025, 1, 1, 12, 0, tzinfo=UTC))
    assert not is_bar_closed(bar_open, "1h", now=datetime(2025, 1, 1, 11, 59, tzinfo=UTC))
