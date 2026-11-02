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
TRADE_KEYS: tuple[str, ...] = (
    "jp_exports",
    "jp_imports",
    "us_exports",
    "us_imports",
    "jp_rate",
)

# OECD publishes Japan's monthly long-term government bond yield with roughly a month's
# delay. A monthly *average* rate is not even knowable until the month ends, so the lag
# runs from the period end like the trade series.
JP_RATE_RELEASE_LAG_DAYS = 35

# How many percentage points of rate differential map to a feature near +/-1. Carry is
# scaled absolutely rather than z-scored, because unlike a momentum signal the *level* is
# what the trade actually earns: a 3-point gap is three points whatever the recent
# distribution of gaps happened to be.
CARRY_SCALE_PP = 2.5

TRADE_FEATURE_NAMES: tuple[str, ...] = (
    "mac_jp_balance",
    "mac_us_balance",
    "mac_balance_diff",
    "mac_jp_import_cost",
    "mac_carry_level",
    "mac_carry_mom",
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


def _carry_features(
    bars: pd.DataFrame,
    *,
    target_tf_ms: int,
    staleness_ms: int,
    us_rate: pd.DataFrame | None,
    jp_rate: pd.DataFrame | None,
    jp_lag: int,
) -> tuple[pd.Series, pd.Series]:
    """Rate-differential carry: ``(level, momentum)``, both bounded to ~[-1, 1].

    The most documented anomaly in foreign exchange, and the one this engine had no way
    to express: hold the higher-yielding currency and earn the gap, until the gap closes
    or the trade unwinds all at once. For USD/JPY the gap is US minus Japan, so a
    positive reading is dollar-supportive.

    The two legs arrive on different clocks and are joined on their own terms — the US
    yield is a daily market rate, known the day it prints; Japan's is a monthly average
    published weeks later. Mixing those would read a figure before it existed.
    """
    neutral = pd.Series(0.0, index=bars.index)
    if us_rate is None or jp_rate is None or len(us_rate) == 0 or len(jp_rate) == 0:
        return neutral, neutral

    day_ms = 86_400_000
    us = pd.DataFrame(
        {
            "ts": us_rate["ts"].astype("int64"),
            "rate": pd.to_numeric(us_rate["close"], errors="coerce"),
        }
    ).dropna()
    us_joined = asof_causal_join(
        bars["ts"],
        us,
        ["rate"],
        target_tf_ms=target_tf_ms,
        ctx_tf_ms=day_ms,
        staleness_limit_ms=staleness_ms,
    )["rate"]

    jp_released = _released_frame(jp_rate, jp_lag)
    if len(jp_released) <= BALANCE_LOOKBACK_MONTHS:
        return neutral, neutral
    jp_derived = pd.DataFrame(
        {
            "ts": jp_released["ts"].astype("int64"),
            "rate": jp_released["value"],
            "chg": jp_released["value"].diff(BALANCE_LOOKBACK_MONTHS),
        }
    )
    jp_joined = asof_causal_join(
        bars["ts"],
        jp_derived,
        ["rate", "chg"],
        target_tf_ms=target_tf_ms,
        ctx_tf_ms=0,
        staleness_limit_ms=staleness_ms,
    )

    # Both legs must be public on the bar, or the "differential" is one leg wearing the
    # name of a comparison — the same per-bar trap the balance differential fell into.
    live = us_joined.notna() & jp_joined["rate"].notna()
    diff = (us_joined - jp_joined["rate"]).where(live)
    level = pd.Series(np.tanh(diff / CARRY_SCALE_PP), index=bars.index).fillna(0.0)

    # Direction of the gap, carried by the Japanese leg's year-over-year change: it moves
    # on a monthly grid and is the slower, more persistent half of the pair. A falling
    # Japanese yield widens the gap, hence the sign.
    widening = _tanh_z_monthly(-jp_joined["chg"]).where(live)
    return level, widening.fillna(0.0)


def build_trade_features(
    bars: pd.DataFrame,
    *,
    timeframe: str,
    frames: dict[str, pd.DataFrame] | None = None,
    lags: dict[str, int] | None = None,
    ctx_frames: dict[str, pd.DataFrame] | None = None,
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
    never = pd.Series(False, index=bars.index)

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
        """Return ``(balance_momentum, import_cost_momentum, available_per_bar)``.

        ``available_per_bar`` is a boolean Series, not a flag: whether this country's
        figures were public is a question with a different answer on every bar, and the
        differential below depends on getting that granularity right.
        """
        exports, imports = released(exports_key), released(imports_key)
        if exports is None or imports is None:
            return neutral, neutral, never
        # Align the two series on their shared availability instants: a balance is only
        # knowable once BOTH sides of it have been published.
        merged = exports.merge(imports, on="ts", how="inner", suffixes=("_x", "_m"))
        if len(merged) <= BALANCE_LOOKBACK_MONTHS:
            return neutral, neutral, never
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
        # NaN out of the join means "nothing published yet", which is emphatically not the
        # same as a published figure that happens to read zero. Keep the distinction.
        return (
            joined["balance"].fillna(0.0),
            joined["import_cost"].fillna(0.0),
            joined["balance"].notna(),
        )

    jp_balance, jp_import_cost, jp_live = balance_feature("jp_exports", "jp_imports")
    us_balance, _, us_live = balance_feature("us_exports", "us_imports")

    # The differential is only a differential on bars where BOTH legs were actually
    # public. Japan's figures run back to 1957 and the US bilateral series only to 1985,
    # so for fourteen years ``us - jp`` would otherwise resolve to a *negated, halved
    # Japan signal* wearing the name of a comparison — non-zero, plausible-looking, and
    # not what it claims to be. Availability is per bar, not per series: a series-level
    # check passes here, because both series do exist; they simply had not both been
    # published yet on the bar being computed.
    diff = ((us_balance - jp_balance) / 2.0).where(jp_live & us_live, 0.0).fillna(0.0)

    carry_level, carry_mom = _carry_features(
        bars,
        target_tf_ms=target_tf_ms,
        staleness_ms=staleness_ms,
        us_rate=(ctx_frames or {}).get("us10y"),
        jp_rate=frames.get("jp_rate"),
        jp_lag=lags.get("jp_rate", JP_RATE_RELEASE_LAG_DAYS),
    )

    features = {
        "mac_jp_balance": jp_balance,
        "mac_us_balance": us_balance,
        # Positive = the US trade position is improving relative to Japan's =
        # dollar-supportive against the yen = USD/JPY up.
        "mac_balance_diff": diff,
        "mac_jp_import_cost": jp_import_cost,
        "mac_carry_level": carry_level,
        "mac_carry_mom": carry_mom,
    }
    out = pd.DataFrame(features, index=bars.index)
    assert list(out.columns) == list(TRADE_FEATURE_NAMES) and len(out) == len(bars)
    return out
