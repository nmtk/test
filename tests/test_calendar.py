from datetime import UTC, date, datetime

from engine.core.calendar import CryptoCalendar, USEquityCalendar, get_calendar
from engine.core.types import AssetClass


def test_crypto_calendar_always_open():
    cal = get_calendar(AssetClass.CRYPTO)
    assert isinstance(cal, CryptoCalendar)
    assert cal.is_open(datetime(2025, 1, 1, 3, 0, tzinfo=UTC))
    assert cal.is_session(date(2025, 1, 1))


def test_us_equity_sessions():
    cal = get_calendar(AssetClass.EQUITY)
    assert isinstance(cal, USEquityCalendar)
    assert cal.is_session(date(2025, 1, 2))  # normal Thursday
    assert not cal.is_session(date(2025, 1, 1))  # New Year's Day
    assert not cal.is_session(date(2025, 1, 4))  # Saturday


def test_us_equity_intraday_open():
    cal = get_calendar(AssetClass.EQUITY)
    # 2025-01-02 15:00 UTC == 10:00 ET (open); 02:00 UTC is overnight (closed).
    assert cal.is_open(datetime(2025, 1, 2, 15, 0, tzinfo=UTC))
    assert not cal.is_open(datetime(2025, 1, 2, 2, 0, tzinfo=UTC))
