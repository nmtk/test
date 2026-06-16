"""Trading calendars.

Crypto trades 24/7; US equities trade only during exchange sessions (and from Japan
that session falls overnight). The executor must respect the right calendar per asset
class, and research must avoid manufacturing bars when a market is closed.

The US equity calendar wraps ``exchange_calendars`` (XNYS) lazily so that importing this
module never requires the dependency unless an equity calendar is actually used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, date, datetime
from functools import cache

from engine.core.types import AssetClass


class TradingCalendar(ABC):
    name: str

    @abstractmethod
    def is_open(self, ts: datetime) -> bool:
        """True if the market is open at instant ``ts`` (tz-aware)."""

    @abstractmethod
    def is_session(self, day: date) -> bool:
        """True if ``day`` is a trading session (ignores intraday hours)."""


class CryptoCalendar(TradingCalendar):
    name = "crypto-24x7"

    def is_open(self, ts: datetime) -> bool:  # noqa: ARG002 - always open
        return True

    def is_session(self, day: date) -> bool:  # noqa: ARG002 - always open
        return True


class USEquityCalendar(TradingCalendar):
    """NYSE regular sessions via ``exchange_calendars`` (XNYS)."""

    name = "XNYS"

    def __init__(self) -> None:
        self._cal = None  # lazily constructed

    def _calendar(self):
        if self._cal is None:
            import exchange_calendars as xcals

            self._cal = xcals.get_calendar("XNYS")
        return self._cal

    def is_open(self, ts: datetime) -> bool:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        # exchange_calendars expects tz-aware UTC timestamps.
        import pandas as pd

        return bool(self._calendar().is_open_on_minute(pd.Timestamp(ts).tz_convert("UTC")))

    def is_session(self, day: date) -> bool:
        import pandas as pd

        return bool(self._calendar().is_session(pd.Timestamp(day)))


@cache
def get_calendar(asset_class: AssetClass) -> TradingCalendar:
    """Return the trading calendar for an asset class (cached singleton)."""
    if asset_class is AssetClass.CRYPTO:
        return CryptoCalendar()
    if asset_class is AssetClass.EQUITY:
        return USEquityCalendar()
    raise ValueError(f"no calendar for asset class {asset_class!r}")
