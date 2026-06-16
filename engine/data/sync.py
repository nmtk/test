"""Incremental OHLCV sync (Job A).

For every universe entry and timeframe, fetch new bars since the last stored bar and
upsert them. Idempotent end to end: re-running after a delay or failure simply converges
the store. The last stored bar is intentionally re-fetched each run so a bar that was
still forming on the previous run gets overwritten with its final, closed values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from engine.core.config import DataSourceSpec, Settings, Universe
from engine.data.sources import make_source
from engine.data.store import OHLCVStore

# How far back to seed history on the very first sync of a series (live venue).
# Deep multi-year research history is backfilled separately via BinanceVisionSource.
_DEFAULT_SEED_DAYS = 365 * 3


@dataclass(slots=True)
class SyncResult:
    source: str
    symbol: str
    timeframe: str
    rows_total: int
    ok: bool
    error: str | None = None


def _seed_since_ms() -> int:
    seed = datetime.now(UTC) - timedelta(days=_DEFAULT_SEED_DAYS)
    return int(seed.timestamp() * 1000)


def sync_series(
    store: OHLCVStore, source_client, spec: DataSourceSpec, timeframe: str
) -> SyncResult:
    """Sync a single (source, symbol, timeframe) series into the store."""
    try:
        last = store.last_ts(spec.source, spec.symbol, timeframe)
        since = last if last is not None else _seed_since_ms()
        df = source_client.fetch_ohlcv(spec.symbol, timeframe, since_ms=since)
        total = store.upsert(spec.source, spec.symbol, timeframe, df)
        return SyncResult(spec.source, spec.symbol, timeframe, total, ok=True)
    except Exception as exc:  # noqa: BLE001 - one bad series must not abort the whole run
        return SyncResult(spec.source, spec.symbol, timeframe, 0, ok=False, error=str(exc))


def sync_universe(
    settings: Settings,
    universe: Universe,
    store: OHLCVStore,
    include_research: bool = False,
) -> list[SyncResult]:
    """Sync every entry's live source (and optionally its research source)."""
    results: list[SyncResult] = []
    source_cache: dict[str, object] = {}

    def client_for(source: str):
        if source not in source_cache:
            source_cache[source] = make_source(source, settings)
        return source_cache[source]

    for entry in universe.entries:
        specs = [entry.live_source]
        if include_research and entry.research_source is not None:
            specs.append(entry.research_source)
        for spec in specs:
            client = client_for(spec.source)
            for timeframe in entry.timeframes:
                results.append(sync_series(store, client, spec, timeframe))
    return results
