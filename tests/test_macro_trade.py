"""Trade-flow feature tests — above all, that released data cannot leak the future."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.features.macro_trade import (
    JP_TRADE_RELEASE_LAG_DAYS,
    TRADE_FEATURE_NAMES,
    US_TRADE_RELEASE_LAG_DAYS,
    availability_ts,
    build_trade_features,
    month_end_ms,
)
from engine.strategies.trade_balance import TradeBalanceUsdJpy

DAY_MS = 86_400_000


def _ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").value // 1_000_000)


def _monthly(start: str, n: int, values) -> pd.DataFrame:
    """A monthly FRED-shaped frame: ``ts`` = period start, ``close`` = reported value."""
    idx = pd.date_range(start, periods=n, freq="MS", tz="UTC")
    ts = (idx.astype("datetime64[ms, UTC]").astype("int64")).to_numpy()
    close = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 0.0}
    )


def _daily_bars(start: str, n: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    ts = (idx.astype("datetime64[ms, UTC]").astype("int64")).to_numpy()
    close = np.linspace(140.0, 155.0, n)
    return pd.DataFrame(
        {"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 0.0}
    )


# -- the release-date model --------------------------------------------------------------


def test_month_end_is_the_start_of_the_next_month():
    jan = pd.Series([_ms("2026-01-01"), _ms("2026-02-01"), _ms("2026-12-01")])
    ends = month_end_ms(jan)
    assert list(ends) == [_ms("2026-02-01"), _ms("2026-03-01"), _ms("2027-01-01")]


def test_availability_is_period_end_plus_publication_lag():
    """January data is public in late February (Japan) / mid March (US), not on Jan 1."""
    january = pd.Series([_ms("2026-01-01")])

    jp = availability_ts(january, JP_TRADE_RELEASE_LAG_DAYS)
    assert jp.iloc[0] == _ms("2026-02-01") + JP_TRADE_RELEASE_LAG_DAYS * DAY_MS
    assert pd.Timestamp(jp.iloc[0], unit="ms", tz="UTC") >= pd.Timestamp("2026-02-20", tz="UTC")

    us = availability_ts(january, US_TRADE_RELEASE_LAG_DAYS)
    # The US FT-900 for January lands in March — over two months after the period stamp.
    assert pd.Timestamp(us.iloc[0], unit="ms", tz="UTC") >= pd.Timestamp("2026-03-05", tz="UTC")
    assert us.iloc[0] > jp.iloc[0]


def test_no_lookahead_features_depend_only_on_already_released_data():
    """The property that matters: truncating the future must not change the past.

    Recomputing the features with every observation released after a cutoff deleted has to
    leave the features at/ before that cutoff bit-for-bit identical. If the join used the
    period stamp instead of the release stamp, the January print would already be visible
    on January 1 and this would fail.
    """
    rng = np.random.default_rng(7)
    n_months = 48
    frames = {
        "jp_exports": _monthly("2022-01-01", n_months, 60 + rng.normal(0, 5, n_months)),
        "jp_imports": _monthly("2022-01-01", n_months, 65 + rng.normal(0, 5, n_months)),
        "us_exports": _monthly("2022-01-01", n_months, 6 + rng.normal(0, 0.5, n_months)),
        "us_imports": _monthly("2022-01-01", n_months, 12 + rng.normal(0, 0.5, n_months)),
    }
    bars = _daily_bars("2022-01-01", 1400)
    full = build_trade_features(bars, timeframe="1d", frames=frames)

    cutoff = _ms("2025-01-01")
    truncated = {
        key: df[
            availability_ts(
                df["ts"],
                JP_TRADE_RELEASE_LAG_DAYS if key.startswith("jp_") else US_TRADE_RELEASE_LAG_DAYS,
            )
            <= cutoff
        ]
        for key, df in frames.items()
    }
    partial = build_trade_features(bars, timeframe="1d", frames=truncated)

    past = bars["ts"] <= cutoff
    assert past.sum() > 500, "need a meaningful stretch of history to compare"
    pd.testing.assert_frame_equal(full[past], partial[past])


def test_period_stamped_join_would_have_leaked():
    """Guards the guard: a naive period-stamp join really does see the future.

    Without this, the test above could pass simply because the data is too smooth to tell
    the two joins apart.
    """
    # 96 months from 2018-01, with a violent regime break in the 2023-01 print. The
    # features need ~36 months of warm-up, so the break sits well clear of it.
    n = 96
    rng = np.random.default_rng(5)
    jitter = rng.normal(0, 0.2, n)  # keeps the rolling z-score's std non-degenerate
    values = np.where(np.arange(n) < 60, 10.0, 99.0) + jitter
    frames = {
        "us_exports": _monthly("2018-01-01", n, values),
        "us_imports": _monthly("2018-01-01", n, 20.0 + jitter),
        "jp_exports": _monthly("2018-01-01", n, 50.0 + jitter),
        "jp_imports": _monthly("2018-01-01", n, 50.0 + jitter),
    }
    bars = _daily_bars("2018-01-01", 2900)

    released = build_trade_features(bars, timeframe="1d", frames=frames)
    # Same data, but pretending each print is public the day its period begins.
    no_lag = build_trade_features(
        bars,
        timeframe="1d",
        frames=frames,
        lags=dict.fromkeys(frames, -31),
    )
    # Between the naive stamp (2023-01-01) and the true US release (~2023-03-12) the two
    # joins must disagree: only the leaky one already knows about the break.
    window = (bars["ts"] >= _ms("2023-01-02")) & (bars["ts"] <= _ms("2023-03-01"))
    assert window.sum() > 30
    assert not np.allclose(
        released.loc[window, "mac_us_balance"], no_lag.loc[window, "mac_us_balance"]
    ), "release lag had no effect — the point-in-time join is not doing its job"
    # And the correctly-lagged features must not have moved yet.
    before = released.loc[bars["ts"] == _ms("2023-01-02"), "mac_us_balance"].iloc[0]
    leaked = no_lag.loc[bars["ts"] == _ms("2023-01-02"), "mac_us_balance"].iloc[0]
    assert abs(leaked) > abs(before)


# -- degradation and bounds --------------------------------------------------------------


def test_missing_series_degrade_to_neutral_not_error():
    bars = _daily_bars("2024-01-01", 200)
    out = build_trade_features(bars, timeframe="1d", frames={})
    assert list(out.columns) == list(TRADE_FEATURE_NAMES)
    assert (out == 0.0).all().all()


def test_one_sided_data_does_not_fake_a_differential():
    """With Japan missing, ``mac_balance_diff`` must stay neutral, not become a US signal."""
    rng = np.random.default_rng(3)
    frames = {
        "us_exports": _monthly("2022-01-01", 48, 6 + rng.normal(0, 0.5, 48)),
        "us_imports": _monthly("2022-01-01", 48, 12 + rng.normal(0, 0.5, 48)),
    }
    bars = _daily_bars("2022-01-01", 1400)
    out = build_trade_features(bars, timeframe="1d", frames=frames)
    assert out["mac_us_balance"].abs().sum() > 0.0, "the US leg should still be computed"
    assert (out["mac_balance_diff"] == 0.0).all()
    assert (out["mac_jp_balance"] == 0.0).all()


def test_features_are_bounded_and_finite():
    rng = np.random.default_rng(11)
    frames = {
        key: _monthly("2020-01-01", 72, 50 + rng.normal(0, 20, 72).cumsum())
        for key in ("jp_exports", "jp_imports", "us_exports", "us_imports")
    }
    bars = _daily_bars("2020-01-01", 2000)
    out = build_trade_features(bars, timeframe="1d", frames=frames)
    assert np.isfinite(out.to_numpy()).all()
    assert out.abs().to_numpy().max() <= 1.0


def test_stale_series_goes_neutral_rather_than_frozen():
    """A series that stops publishing must not pin the signal forever."""
    frames = {
        key: _monthly("2020-01-01", 30, np.linspace(10, 40, 30))
        for key in ("jp_exports", "us_exports")
    } | {
        key: _monthly("2020-01-01", 30, np.linspace(40, 10, 30))
        for key in ("jp_imports", "us_imports")
    }
    bars = _daily_bars("2020-01-01", 2400)  # runs years past the last observation
    out = build_trade_features(bars, timeframe="1d", frames=frames)
    assert (out.iloc[-1] == 0.0).all()


# -- the strategy ------------------------------------------------------------------------


def test_strategy_is_flat_without_features():
    bars = _daily_bars("2024-01-01", 300)
    weights = TradeBalanceUsdJpy().weights(bars)
    assert (weights == 0.0).all()
    assert weights.index.equals(bars.index)


def test_strategy_direction_flips_the_book():
    bars = _daily_bars("2024-01-01", 300)
    bars = bars.assign(mac_balance_diff=0.5)
    long_book = TradeBalanceUsdJpy(direction=1, threshold=0.1, smooth=1).weights(bars)
    short_book = TradeBalanceUsdJpy(direction=-1, threshold=0.1, smooth=1).weights(bars)
    assert (long_book == 1.0).all()
    assert (short_book == -1.0).all()


def test_strategy_respects_threshold_and_bounds():
    bars = _daily_bars("2024-01-01", 300).assign(mac_balance_diff=0.05)
    assert (TradeBalanceUsdJpy(threshold=0.2, smooth=1).weights(bars) == 0.0).all()
    weights = TradeBalanceUsdJpy(threshold=0.0, smooth=1).weights(bars)
    assert weights.abs().max() <= 1.0


def test_strategy_rejects_unknown_signal():
    with pytest.raises(ValueError, match="unknown signal"):
        TradeBalanceUsdJpy(signal="not_a_signal")


def test_differential_is_neutral_while_either_leg_is_unpublished():
    """Availability is per *bar*, not per series — the deep-history bug.

    Japan's trade figures run back to 1957 and the US bilateral series only to 1985. Both
    series exist, so a series-level check passes, yet for fourteen years only one leg had
    actually been published on any given bar. ``(us - jp)/2`` then resolves to a negated,
    halved Japan signal wearing the name of a comparison: non-zero, plausible, wrong.
    """
    rng = np.random.default_rng(19)
    long_n, short_n = 240, 120  # Japan starts 10 years before the US
    frames = {
        "jp_exports": _monthly("2000-01-01", long_n, 60 + rng.normal(0, 5, long_n)),
        "jp_imports": _monthly("2000-01-01", long_n, 65 + rng.normal(0, 5, long_n)),
        "us_exports": _monthly("2010-01-01", short_n, 6 + rng.normal(0, 0.5, short_n)),
        "us_imports": _monthly("2010-01-01", short_n, 12 + rng.normal(0, 0.5, short_n)),
    }
    bars = _daily_bars("2000-01-01", 7000)
    out = build_trade_features(bars, timeframe="1d", frames=frames)

    early = bars["ts"] < _ms("2009-01-01")
    assert out.loc[early, "mac_jp_balance"].abs().sum() > 0.0, "Japan's leg should be live"
    assert (out.loc[early, "mac_us_balance"] == 0.0).all(), "the US leg is not published yet"
    assert (out.loc[early, "mac_balance_diff"] == 0.0).all(), (
        "a differential with one unpublished leg must read neutral, not half the other leg"
    )

    # Once both legs are public the differential switches on.
    late = bars["ts"] > _ms("2015-01-01")
    assert out.loc[late, "mac_balance_diff"].abs().sum() > 0.0


def test_differential_matches_the_legs_where_both_are_live():
    rng = np.random.default_rng(23)
    frames = {
        key: _monthly("2000-01-01", 240, 50 + rng.normal(0, 4, 240))
        for key in ("jp_exports", "jp_imports", "us_exports", "us_imports")
    }
    bars = _daily_bars("2000-01-01", 7000)
    out = build_trade_features(bars, timeframe="1d", frames=frames)
    live = out["mac_balance_diff"] != 0.0
    assert live.sum() > 100
    expected = (out.loc[live, "mac_us_balance"] - out.loc[live, "mac_jp_balance"]) / 2.0
    pd.testing.assert_series_equal(out.loc[live, "mac_balance_diff"], expected, check_names=False)
