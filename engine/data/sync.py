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
# Dukascopy is here for a different reason than bitbank: it serves one file per
# instrument-hour, so fetching 5m, 1h and 1d natively would triple an already large
# request budget for data that is identical after resampling.
DERIVE_SOURCES = {"bitbank", "dukascopy"}


@dataclass(slots=True)
class SyncResult:
    source: str
    symbol: str
    timeframe: str
    rows_total: int
    ok: bool
    error: str | None = None


# Timeframes holding released statistics rather than bars. These are always fetched in
# full rather than incrementally — see :func:`sync_series`.
FULL_REFETCH_TIMEFRAMES = {"1mo"}


def _seed_since_ms() -> int:
    seed = datetime.now(UTC) - timedelta(days=_DEFAULT_SEED_DAYS)
    return int(seed.timestamp() * 1000)


def sync_series(
    store: OHLCVStore, source_client, spec: DataSourceSpec, timeframe: str
) -> SyncResult:
    """Sync a single (source, symbol, timeframe) series natively into the store.

    Monthly released statistics are re-fetched **whole** every run, for two reasons that
    both bit in production:

    * *Depth.* The endpoint returns the entire series in one request either way, so the
      incremental seed was pure loss: a 3-year seed capped the trade-flow series at 34
      monthly prints, and the feature layer needs 36 (a 12-month difference, z-scored over
      a window needing 24 points) before it emits a single non-neutral value. The
      strategy was inert and nothing said so.
    * *Revisions.* Trade statistics are revised after first publication. An incremental
      sync anchored on the last timestamp would never see a corrected earlier month.

    Caveat this does not fix: using the *current* vintage of a revised figure is a mild
    look-ahead, since the backtest reads a number nobody had on the release date. Fixing
    that properly needs vintage data (FRED's ALFRED). The release-lag model in
    engine/features/macro_trade.py handles *when* a figure appeared, not which revision
    of it — the far larger of the two errors, but not the only one.
    """
    try:
        if timeframe in FULL_REFETCH_TIMEFRAMES:
            since = None
        else:
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


def sync_context(settings: Settings, store: OHLCVStore) -> list[SyncResult]:
    """Sync every configured cross-asset context series (macro flow inputs).

    Reads ``strategies_config/context.yaml``; an absent file is simply "no context".
    Same idempotent upsert path as instrument bars, stored under
    ``data/ohlcv/{source}/{symbol}/{tf}.parquet``.
    """
    from engine.core.config import load_context_series

    series = load_context_series(settings.context_path)
    results: list[SyncResult] = []
    source_cache: dict[str, object] = {}
    for spec in series:
        if spec.source not in source_cache:
            source_cache[spec.source] = make_source(spec.source, settings)
        data_spec = DataSourceSpec(source=spec.source, symbol=spec.symbol)
        results.append(sync_series(store, source_cache[spec.source], data_spec, spec.timeframe))
    return results
