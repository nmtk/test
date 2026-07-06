"""Causal cross-asset context features (the "flow" layer).

Answers one question for every target bar: *given everything already closed elsewhere —
stocks, gold, rates, FX, and BTC itself — is money likely flowing into or out of this
instrument?* All features are bounded to roughly [-1, 1] so they compose cleanly in GP
programs and give the ``macro_flow`` archetype a uniform threshold scale.

Point-in-time rule (the part that must never be wrong): a context bar is visible to a
target bar only when the context bar's *close time* is at or before the target bar's
close time::

    ctx.ts + ctx_tf  <=  target.ts + target_tf

Daily macro closes therefore only affect target bars from the next UTC day onward (a US
equity close at 21:00 UTC becomes visible at 00:00 UTC — conservatively late, never
early). On top of that, the backtester shifts weights one further bar before applying
them, so there are two independent layers against look-ahead.

Missing data degrades to *neutral* (0.0) features, never to an error: strategies keep
running (flat on context-gated signals) when a macro source is down or not yet synced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.core.clock import timeframe_seconds
from engine.core.config import ContextSeries, Settings, Universe, load_context_series
from engine.core.types import AssetClass
from engine.data.store import OHLCVStore

# The macro series keys the feature layer understands. Each key contributes
# ``ctx_{key}_mom`` and ``ctx_{key}_trend``; the composites below consume specific keys.
# Extending the GP/archetype feature space = add a key here + a row in
# strategies_config/context.yaml (see docs/STRATEGY.md).
CONTEXT_KEYS: tuple[str, ...] = ("spx", "gold", "us10y", "usdjpy", "eurusd")

# Composite + crypto-market-factor features (computed from the keys / BTC bars).
_COMPOSITE_NAMES: tuple[str, ...] = ("ctx_dollar_mom", "ctx_risk_on")
_MARKET_NAMES: tuple[str, ...] = ("ctx_btc_mom", "ctx_rel_btc_mom", "ctx_btc_corr")

CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    *(f"ctx_{key}_{kind}" for key in CONTEXT_KEYS for kind in ("mom", "trend")),
    *_COMPOSITE_NAMES,
    *_MARKET_NAMES,
)

MOM_LOOKBACK = 21  # bars of the context series' own timeframe
TREND_LOOKBACK = 63
MOM_NORM_WINDOW = 252  # rolling window used to z-scale momentum before tanh
CORR_LOOKBACK = 63  # target-timeframe bars for the target-vs-BTC return correlation
STALENESS_LIMIT_BARS = 10  # context older than this many of its own bars -> neutral


def _tanh_z(change: pd.Series) -> pd.Series:
    """Squash an n-bar change to (-1, 1) via a rolling z-score. Scale-free and causal."""
    std = change.rolling(MOM_NORM_WINDOW, min_periods=MOM_LOOKBACK * 2).std(ddof=0)
    z = change / std.replace(0.0, np.nan)
    return pd.Series(np.tanh(z), index=change.index)


def _mom(close: pd.Series, lookback: int = MOM_LOOKBACK) -> pd.Series:
    """Bounded momentum: tanh of the z-scored ``lookback``-bar log return."""
    logret = np.log(close.replace(0.0, np.nan)).diff(lookback)
    return _tanh_z(logret)


def _trend(close: pd.Series, lookback: int = TREND_LOOKBACK) -> pd.Series:
    """Sign of close minus its ``lookback``-bar SMA: {-1, 0, +1}."""
    sma = close.rolling(lookback, min_periods=lookback // 2).mean()
    return pd.Series(np.sign(close - sma), index=close.index)


def asof_causal_join(
    target_ts: pd.Series,
    ctx: pd.DataFrame,
    columns: list[str],
    *,
    target_tf_ms: int,
    ctx_tf_ms: int,
    staleness_limit_ms: int | None = None,
) -> pd.DataFrame:
    """Join context ``columns`` onto target bars using the close-time visibility rule.

    ``ctx`` must carry an ascending int-ms ``ts`` column (bar open times). Returns a frame
    aligned to ``target_ts`` (same length/order); rows with no visible or non-stale
    context are NaN.
    """
    avail = ctx["ts"].to_numpy(dtype="int64") + ctx_tf_ms  # when each ctx bar closes
    decision = target_ts.to_numpy(dtype="int64") + target_tf_ms  # when each target bar closes
    idx = np.searchsorted(avail, decision, side="right") - 1
    valid = idx >= 0
    if staleness_limit_ms is not None:
        age = decision - np.where(valid, avail[np.clip(idx, 0, None)], 0)
        valid &= age <= staleness_limit_ms
    out = {}
    safe_idx = np.clip(idx, 0, max(len(ctx) - 1, 0))
    for col in columns:
        values = ctx[col].to_numpy(dtype=float)[safe_idx] if len(ctx) else np.array([])
        out[col] = np.where(valid, values, np.nan) if len(ctx) else np.full(len(idx), np.nan)
    return pd.DataFrame(out, index=target_ts.index)


def build_context_features(
    bars: pd.DataFrame,
    *,
    timeframe: str,
    ctx_frames: dict[str, pd.DataFrame] | None = None,
    market_bars: pd.DataFrame | None = None,
    market_is_self: bool = False,
) -> pd.DataFrame:
    """Compute every ``ctx_*`` feature column for ``bars`` (all causal, all in ~[-1, 1]).

    ``ctx_frames`` maps a :data:`CONTEXT_KEYS` key to a daily OHLCV frame; missing keys
    yield neutral (0.0) features. ``market_bars`` is the BTC series at the *same*
    timeframe/source as the target (pass ``market_is_self=True`` when the target IS the
    market factor, so relative features are exactly neutral rather than noise).
    """
    ctx_frames = ctx_frames or {}
    target_tf_ms = timeframe_seconds(timeframe) * 1000
    day_ms = timeframe_seconds("1d") * 1000
    n = len(bars)
    features: dict[str, pd.Series] = {}

    per_key: dict[str, pd.DataFrame] = {}
    for key in CONTEXT_KEYS:
        raw = ctx_frames.get(key)
        if raw is None or len(raw) == 0:
            continue
        close = raw["close"].astype(float)
        derived = pd.DataFrame(
            {"ts": raw["ts"].astype("int64"), "mom": _mom(close), "trend": _trend(close)}
        )
        per_key[key] = derived

    for key in CONTEXT_KEYS:
        for kind in ("mom", "trend"):
            name = f"ctx_{key}_{kind}"
            derived = per_key.get(key)
            if derived is None:
                features[name] = pd.Series(0.0, index=bars.index)
                continue
            joined = asof_causal_join(
                bars["ts"],
                derived,
                [kind],
                target_tf_ms=target_tf_ms,
                ctx_tf_ms=day_ms,
                staleness_limit_ms=STALENESS_LIMIT_BARS * day_ms,
            )
            features[name] = joined[kind].fillna(0.0)

    # Dollar strength: USD up vs both JPY and EUR = global liquidity draining out of
    # risk assets. (eurusd is USD-per-EUR, so dollar strength = -eurusd momentum.)
    features["ctx_dollar_mom"] = (
        (features["ctx_usdjpy_mom"] - features["ctx_eurusd_mom"]) / 2.0
    ).fillna(0.0)

    # Risk-on composite in [-1, 1]: equities up, dollar not bid, gold not bid.
    features["ctx_risk_on"] = (
        (
            np.sign(features["ctx_spx_mom"])
            - np.sign(features["ctx_dollar_mom"])
            - np.sign(features["ctx_gold_mom"])
        )
        / 3.0
    ).fillna(0.0)

    # Crypto market factor: BTC momentum, target-vs-BTC relative momentum (rotation), and
    # the rolling correlation of target vs BTC returns (how coupled the target is).
    close = bars["close"].astype(float)
    if market_is_self:
        features["ctx_btc_mom"] = _mom(close).fillna(0.0)
        features["ctx_rel_btc_mom"] = pd.Series(0.0, index=bars.index)
        features["ctx_btc_corr"] = pd.Series(1.0, index=bars.index)
    elif market_bars is not None and len(market_bars) >= MOM_LOOKBACK:
        mkt_close = market_bars["close"].astype(float)
        mkt = pd.DataFrame(
            {
                "ts": market_bars["ts"].astype("int64"),
                "mom": _mom(mkt_close),
                "logret": np.log(mkt_close.replace(0.0, np.nan)).diff(),
                "log21": np.log(mkt_close.replace(0.0, np.nan)).diff(MOM_LOOKBACK),
            }
        )
        joined = asof_causal_join(
            bars["ts"],
            mkt,
            ["mom", "logret", "log21"],
            target_tf_ms=target_tf_ms,
            ctx_tf_ms=target_tf_ms,
            staleness_limit_ms=STALENESS_LIMIT_BARS * target_tf_ms,
        )
        features["ctx_btc_mom"] = joined["mom"].fillna(0.0)
        own_log21 = np.log(close.replace(0.0, np.nan)).diff(MOM_LOOKBACK)
        features["ctx_rel_btc_mom"] = _tanh_z(own_log21 - joined["log21"]).fillna(0.0)
        own_ret = close.pct_change()
        corr = own_ret.rolling(CORR_LOOKBACK, min_periods=CORR_LOOKBACK // 2).corr(joined["logret"])
        # A near-constant window makes pandas' rolling corr numerically explode; a true
        # correlation is bounded, so clip after neutralizing non-finite values.
        features["ctx_btc_corr"] = (
            corr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        )
    else:
        features["ctx_btc_mom"] = pd.Series(0.0, index=bars.index)
        features["ctx_rel_btc_mom"] = pd.Series(0.0, index=bars.index)
        features["ctx_btc_corr"] = pd.Series(0.0, index=bars.index)

    out = pd.DataFrame(features, index=bars.index)
    assert list(out.columns) == list(CONTEXT_FEATURE_NAMES) and len(out) == n
    return out


@dataclass(slots=True)
class ContextAugmenter:
    """Loads context data once and stamps ``ctx_*`` columns onto any target's bars.

    Used identically by research, the paper executor, and the live executor, so a
    strategy validated on context features sees the *same* features in production.
    """

    store: OHLCVStore
    series: list[ContextSeries]
    market_by_source: dict[str, str]  # data source -> BTC symbol on that source
    _frames: dict[str, pd.DataFrame] | None = None

    @classmethod
    def from_settings(cls, settings: Settings, universe: Universe | None = None):
        series = load_context_series(settings.context_path)
        market: dict[str, str] = {}
        if universe is not None:
            for entry in universe.by_asset_class(AssetClass.CRYPTO):
                if (entry.instrument.base or "").upper() != "BTC":
                    continue
                market.setdefault(entry.live_source.source, entry.live_source.symbol)
                if entry.research_source is not None:
                    market.setdefault(entry.research_source.source, entry.research_source.symbol)
        return cls(
            store=OHLCVStore(settings.data_dir / "ohlcv"), series=series, market_by_source=market
        )

    def _context_frames(self) -> dict[str, pd.DataFrame]:
        if self._frames is None:
            self._frames = {}
            for spec in self.series:
                if spec.key not in CONTEXT_KEYS:
                    continue  # unknown keys are synced but not featurized (see CONTEXT_KEYS)
                df = self.store.read(spec.source, spec.symbol, spec.timeframe)
                if len(df):
                    self._frames[spec.key] = df
        return self._frames

    def augment(
        self, bars: pd.DataFrame, *, source: str, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """Return ``bars`` plus every ``ctx_*`` column (neutral where data is missing)."""
        if len(bars) == 0:
            return bars
        market_symbol = self.market_by_source.get(source)
        market_is_self = market_symbol == symbol
        market_bars = None
        if market_symbol and not market_is_self:
            market_bars = self.store.read(source, market_symbol, timeframe)
        ctx = build_context_features(
            bars,
            timeframe=timeframe,
            ctx_frames=self._context_frames(),
            market_bars=market_bars,
            market_is_self=market_is_self,
        )
        out = bars.copy()
        for col in ctx.columns:
            out[col] = ctx[col]
        return out
