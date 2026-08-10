"""Point-in-time trade-flow features for USD/JPY (Japan vs US import/export data).

The economic hypothesis: a country's merchandise trade balance is *real, non-speculative
currency demand*. Japanese exporters repatriating USD revenue bid the yen; Japanese
importers — above all of energy, which Japan buys in dollars — sell yen to pay for it.
When Japan's balance deteriorates, that structural flow turns into persistent yen supply.
The mirror argument applies to the US. So the **differential** between the two balances is
a slow-moving fundamental pressure on USD/JPY, distinct from the rate-differential and
risk-sentiment stories the rest of the engine already conditions on.

Why this module exists rather than a line in ``context.yaml``
-------------------------------------------------------------
Every other context series in this engine is a *market price*: it is known the instant it
prints. Trade statistics are not. They describe a month that is already over, and they are
published weeks later:

* Japan (財務省貿易統計, MoF): preliminary figures for month *M* land around the 20th of
  month *M+1*.
* United States (Census/BEA FT-900): month *M* lands around the 5th business day of month
  *M+2* — roughly five weeks after the period ends.

FRED stamps each observation with the **first day of the period it describes**, not its
release date. Joining on that stamp would let a backtest read January's trade balance on
January 1st — over two months of look-ahead on the US series, which is more than enough to
manufacture a spectacular and completely fictitious backtest. This module therefore maps
every observation to the date it *became public* and joins on that instead.

The lags below are deliberately **conservative** (later than the true release): being late
costs a little edge, being early invents it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.core.clock import timeframe_seconds
from engine.features.context import asof_causal_join

# Publication lag in days measured from the END of the observation period, rounded up from
# the publishers' own calendars. See the module docstring for sources.
JP_TRADE_RELEASE_LAG_DAYS = 25  # MoF preliminary ~20th of M+1
US_TRADE_RELEASE_LAG_DAYS = 40  # Census/BEA FT-900 ~5th business day of M+2

# Series keys this layer understands. Missing keys degrade to neutral features, never to an
# error — same contract as the cross-asset context layer.
TRADE_KEYS: tuple[str, ...] = ("jp_exports", "jp_imports", "us_exports", "us_imports")

TRADE_FEATURE_NAMES: tuple[str, ...] = (
    "mac_jp_balance",
    "mac_us_balance",
    "mac_balance_diff",
    "mac_jp_import_cost",
)

BALANCE_LOOKBACK_MONTHS = 12  # year-over-year change kills the seasonality in trade data
STALENESS_LIMIT_DAYS = 200  # a series that stops publishing goes neutral rather than stale

# Normalization window for the monthly z-scale. The cross-asset context layer normalizes
# over 252 observations because its series are *daily* (252 bars = one year); reusing that
# figure here would demand 21 years of monthly prints before a single feature turned on.
# Five years of history, usable from two, is the monthly equivalent.
NORM_WINDOW_MONTHS = 60
NORM_MIN_MONTHS = 24


def _tanh_z_monthly(change: pd.Series) -> pd.Series:
    """Squash a monthly change to (-1, 1) via a rolling z-score. Scale-free and causal."""
    std = change.rolling(NORM_WINDOW_MONTHS, min_periods=NORM_MIN_MONTHS).std(ddof=0)
    z = change / std.replace(0.0, np.nan)
    return pd.Series(np.tanh(z), index=change.index)


def _safe_log(values: pd.Series) -> pd.Series:
    """Log of strictly positive values; anything else is NaN rather than a warning."""
    positive = values.where(values > 0.0)
    return pd.Series(np.log(positive.to_numpy(dtype=float)), index=values.index)


@dataclass(frozen=True, slots=True)
class TradeSeriesSpec:
    """One macro trade series plus the lag between its period end and its publication."""

    key: str
    source: str
    symbol: str
    release_lag_days: int


DEFAULT_TRADE_SERIES: tuple[TradeSeriesSpec, ...] = (
    # US Census bilateral goods trade with Japan — the flow that most directly moves
    # USD/JPY, and a direct read on both countries' import/export performance.
    TradeSeriesSpec("us_exports", "fred", "EXPJP", US_TRADE_RELEASE_LAG_DAYS),
    TradeSeriesSpec("us_imports", "fred", "IMPJP", US_TRADE_RELEASE_LAG_DAYS),
    # Japan's total goods trade (OECD/MoF via FRED). If FRED retires these IDs the
    # features degrade to neutral and the strategy stays flat — see module docstring.
    TradeSeriesSpec("jp_exports", "fred", "XTEXVA01JPM667S", JP_TRADE_RELEASE_LAG_DAYS),
    TradeSeriesSpec("jp_imports", "fred", "XTIMVA01JPM667S", JP_TRADE_RELEASE_LAG_DAYS),
)


def month_end_ms(period_start_ms: pd.Series) -> pd.Series:
    """Map a monthly observation's period-start stamp to the instant the period ends."""
    starts = pd.to_datetime(period_start_ms.astype("int64"), unit="ms", utc=True)
    ends = starts + pd.offsets.MonthBegin(1)
    return pd.Series(
        ends.astype("datetime64[ms, UTC]").astype("int64"), index=period_start_ms.index
    )


def availability_ts(period_start_ms: pd.Series, release_lag_days: int) -> pd.Series:
    """Epoch-ms instant at which an observation becomes public knowledge.

    ``period_end + release_lag_days``. This is the only timestamp the feature layer is
    allowed to join on; the raw observation stamp describes the past, not the present.
    """
    lag_ms = int(release_lag_days) * 86_400_000
    return month_end_ms(period_start_ms) + lag_ms


def _released_frame(raw: pd.DataFrame, release_lag_days: int) -> pd.DataFrame:
    """Return ``ts`` (= availability instant) + ``value``, ascending, deduplicated.

    Emitting the availability instant as ``ts`` lets the join reuse
    :func:`engine.features.context.asof_causal_join` with ``ctx_tf_ms=0``.
    """
    out = pd.DataFrame(
        {
            "ts": availability_ts(raw["ts"], release_lag_days),
            "value": pd.to_numeric(raw["close"], errors="coerce"),
        }
    )
    return out.dropna(subset=["value"]).drop_duplicates(subset="ts", keep="last").sort_values("ts")


def _normalized_balance(exports: pd.Series, imports: pd.Series) -> pd.Series:
    """Balance as a share of total trade: ``(X - M) / (X + M)``, in ``[-1, 1]``.

    Scale-free, so it is comparable across countries reporting in different currencies and
    magnitudes (yen vs dollars) and immune to the long-run growth of nominal trade.
    """
    total = (exports + imports).replace(0.0, np.nan)
    return ((exports - imports) / total).replace([np.inf, -np.inf], np.nan)


def build_trade_features(
    bars: pd.DataFrame,
    *,
    timeframe: str,
    frames: dict[str, pd.DataFrame] | None = None,
    lags: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute every ``mac_*`` trade-flow feature for ``bars`` (causal, all in ~[-1, 1]).

    ``frames`` maps a :data:`TRADE_KEYS` key to the raw monthly OHLCV frame as stored by
    the sync job (``ts`` = period start, ``close`` = the reported value). Any missing or
    unusable key yields neutral (0.0) features so the book stays flat rather than trading
    an input it cannot see.
    """
    frames = frames or {}
    lags = lags or {}
    target_tf_ms = timeframe_seconds(timeframe) * 1000
    staleness_ms = STALENESS_LIMIT_DAYS * 86_400_000
    neutral = pd.Series(0.0, index=bars.index)

    def released(key: str) -> pd.DataFrame | None:
        raw = frames.get(key)
        if raw is None or len(raw) == 0:
            return None
        lag = lags.get(
            key,
            JP_TRADE_RELEASE_LAG_DAYS if key.startswith("jp_") else US_TRADE_RELEASE_LAG_DAYS,
        )
        frame = _released_frame(raw, lag)
        return frame if len(frame) else None

    def balance_feature(exports_key: str, imports_key: str) -> tuple[pd.Series, pd.Series, bool]:
        """Return ``(balance_momentum, import_cost_momentum, available)`` for the bars.

        ``available`` is False whenever the pair could not be computed, so callers can tell
        a genuine neutral reading apart from missing data.
        """
        exports, imports = released(exports_key), released(imports_key)
        if exports is None or imports is None:
            return neutral, neutral, False
        # Align the two series on their shared availability instants: a balance is only
        # knowable once BOTH sides of it have been published.
        merged = exports.merge(imports, on="ts", how="inner", suffixes=("_x", "_m"))
        if len(merged) <= BALANCE_LOOKBACK_MONTHS:
            return neutral, neutral, False
        balance = _normalized_balance(merged["value_x"], merged["value_m"])
        derived = pd.DataFrame(
            {
                "ts": merged["ts"].astype("int64"),
                # Year-over-year *change* in the balance: is the flow improving or
                # deteriorating? Levels are dominated by structural constants.
                "balance": _tanh_z_monthly(balance.diff(BALANCE_LOOKBACK_MONTHS)),
                # Import cost surge — Japan's energy-import channel specifically.
                "import_cost": _tanh_z_monthly(
                    _safe_log(merged["value_m"]).diff(BALANCE_LOOKBACK_MONTHS)
                ),
            }
        ).dropna(subset=["ts"])
        joined = asof_causal_join(
            bars["ts"],
            derived,
            ["balance", "import_cost"],
            target_tf_ms=target_tf_ms,
            ctx_tf_ms=0,  # ``ts`` is already the availability instant
            staleness_limit_ms=staleness_ms,
        )
        return joined["balance"].fillna(0.0), joined["import_cost"].fillna(0.0), True

    jp_balance, jp_import_cost, jp_ok = balance_feature("jp_exports", "jp_imports")
    us_balance, _, us_ok = balance_feature("us_exports", "us_imports")

    # The differential is only a differential when BOTH legs exist. With one country
    # missing, ``us - jp`` would quietly collapse to a single-country signal wearing the
    # name of a comparison — so it reads neutral instead and the strategy stays flat.
    diff = ((us_balance - jp_balance) / 2.0).fillna(0.0) if (jp_ok and us_ok) else neutral

    features = {
        "mac_jp_balance": jp_balance,
        "mac_us_balance": us_balance,
        # Positive = the US trade position is improving relative to Japan's =
        # dollar-supportive against the yen = USD/JPY up.
        "mac_balance_diff": diff,
        "mac_jp_import_cost": jp_import_cost,
    }
    out = pd.DataFrame(features, index=bars.index)
    assert list(out.columns) == list(TRADE_FEATURE_NAMES) and len(out) == len(bars)
    return out
