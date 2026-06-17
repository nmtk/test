"""Incremental OHLCV sync (Job A).

For every universe entry and timeframe, fetch new bars since the last stored bar and
upsert them. Idempotent end to end: re-running after a delay or failure simply converges
the store. The last stored bar is intentionally re-fetched each run so a bar that was
still forming on the previous run gets overwritten with its final, closed values.

Some venues can't serve every timeframe reliably (bitbank's ccxt 4h/1d return error
10000), so for those we fetch only the base (smallest) timeframe and *derive* the larger
ones by resampling — one consistent source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from engine.core.clock import timeframe_seconds
from engine.core.config import DataSourceSpec, Settings, Universe
from engine.data.resample import resample_ohlcv
from engine.data.sources import make_source
from engine.data.store import OHLCVStore

# How far back to seed history on the very first sync of a series (live venue).
# Deep multi-year research history is backfilled separately via BinanceVisionSource.
_DEFAULT_SEED_DAYS = 365 * 3

# Sources whose higher timeframes are derived from the base timeframe by resampling.
DERIVE_SOURCES = {"bitbank"}


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
    """Sync a single (source, symbol, timeframe) series natively into the store."""
    try:
        last = store.last_ts(spec.source, spec.symbol, timeframe)
        since = last if last is not None else _seed_since_ms()
        df = source_client.fetch_ohlcv(spec.symbol, timeframe, since_ms=since)
        total = store.upsert(spec.source, spec.symbol, timeframe, df)
        return SyncResult(spec.source, spec.symbol, timeframe, total, ok=True)
    except Exception as exc:  # noqa: BLE001 - one bad series must not abort the whole run
        return SyncResult(spec.source, spec.symbol, timeframe, 0, ok=False, error=str(exc))


def derive_series(
    store: OHLCVStore, source: str, symbol: str, base_tf: str, target_tf: str
) -> SyncResult:
    """Derive a higher timeframe by resampling the stored base timeframe."""
    try:
        base = store.read(source, symbol, base_tf)
        if len(base) == 0:
            return SyncResult(source, symbol, target_tf, 0, ok=False, error="no base data")
        resampled = resample_ohlcv(base, base_tf, target_tf)
        total = store.upsert(source, symbol, target_tf, resampled)
        return SyncResult(source, symbol, target_tf, total, ok=True)
    except Exception as exc:  # noqa: BLE001
        return SyncResult(source, symbol, target_tf, 0, ok=False, error=str(exc))


def _sync_spec(
    store: OHLCVStore, client, spec: DataSourceSpec, timeframes: tuple[str, ...]
) -> list[SyncResult]:
    if spec.source in DERIVE_SOURCES and len(timeframes) > 1:
        base_tf = min(timeframes, key=timeframe_seconds)
        results = [sync_series(store, client, spec, base_tf)]
        for tf in timeframes:
            if tf != base_tf:
                results.append(derive_series(store, spec.source, spec.symbol, base_tf, tf))
        return results
    return [sync_series(store, client, spec, tf) for tf in timeframes]


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
            results.extend(_sync_spec(store, client_for(spec.source), spec, entry.timeframes))
    return results
