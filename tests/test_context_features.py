"""Context (cross-asset flow) feature layer: causality, bounds, and graceful degradation."""

import numpy as np
import pandas as pd

from engine.data.store import OHLCVStore
from engine.features.context import (
    CONTEXT_FEATURE_NAMES,
    ContextAugmenter,
    asof_causal_join,
    build_context_features,
)

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
T0 = 1_600_000_000_000  # aligned to a UTC day boundary? not needed; join is pure arithmetic


def _ohlcv(ts, close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "ts": np.asarray(ts, dtype="int64"),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
        }
    )


def _daily_ctx(n=400, start=T0, values=None):
    ts = start + np.arange(n) * DAY_MS
    close = values if values is not None else 100.0 + np.sin(np.arange(n) / 7.0)
    return _ohlcv(ts, close)


def _hourly_bars(n=500, start=None):
    start = start if start is not None else T0 + 300 * DAY_MS
    ts = start + np.arange(n) * HOUR_MS
    close = 100.0 * np.cumprod(1.0 + 0.001 * np.sin(np.arange(n) / 5.0))
    return _ohlcv(ts, close)


# -- the visibility rule ------------------------------------------------------------------


def test_asof_join_respects_close_time_boundary():
    # One daily ctx bar opening at T0 closes at T0 + 1d.
    ctx = pd.DataFrame({"ts": [T0], "x": [42.0]})
    # Hourly target bars: close times T0+1d-1h (must NOT see it) and T0+1d (must see it).
    target_ts = pd.Series([T0 + DAY_MS - 2 * HOUR_MS, T0 + DAY_MS - HOUR_MS], dtype="int64")
    joined = asof_causal_join(target_ts, ctx, ["x"], target_tf_ms=HOUR_MS, ctx_tf_ms=DAY_MS)
    assert np.isnan(joined["x"].iloc[0])
    assert joined["x"].iloc[1] == 42.0


def test_asof_join_staleness_cap_goes_neutral():
    ctx = pd.DataFrame({"ts": [T0], "x": [42.0]})
    fresh = T0 + 2 * DAY_MS
    stale = T0 + 40 * DAY_MS
    joined = asof_causal_join(
        pd.Series([fresh, stale], dtype="int64"),
        ctx,
        ["x"],
        target_tf_ms=HOUR_MS,
        ctx_tf_ms=DAY_MS,
        staleness_limit_ms=10 * DAY_MS,
    )
    assert joined["x"].iloc[0] == 42.0
    assert np.isnan(joined["x"].iloc[1])


def test_context_jump_is_not_visible_early():
    # SPX doubles on day D. No target bar closing on/before D's close may reflect it.
    n_days = 400
    jump_day = 350
    values = np.full(n_days, 100.0)
    values[jump_day:] = 200.0
    ctx = {"spx": _daily_ctx(n=n_days, values=values)}
    jump_open_ts = T0 + jump_day * DAY_MS

    bars = _hourly_bars(n=24 * 60, start=jump_open_ts - 30 * DAY_MS)
    feats = build_context_features(bars, timeframe="1h", ctx_frames=ctx)

    visible_from = jump_open_ts + DAY_MS  # the jump bar's close time
    before = feats.loc[bars["ts"] + HOUR_MS < visible_from, "ctx_spx_mom"]
    after = feats.loc[bars["ts"] + HOUR_MS >= visible_from, "ctx_spx_mom"]
    # Before visibility the momentum is ~0 (flat series); after it saturates positive.
    assert before.abs().max() < 0.01
    assert after.max() > 0.9


def test_features_are_append_invariant():
    # Extending the context series with future data must not change past features.
    ctx_full = {"spx": _daily_ctx(n=400)}
    ctx_short = {"spx": _daily_ctx(n=360)}
    bars = _hourly_bars(n=300, start=T0 + 340 * DAY_MS)

    cutoff = T0 + 360 * DAY_MS  # ctx_short ends here; compare bars fully covered by both
    mask = (bars["ts"] + HOUR_MS) <= cutoff
    f_full = build_context_features(bars, timeframe="1h", ctx_frames=ctx_full)
    f_short = build_context_features(bars, timeframe="1h", ctx_frames=ctx_short)
    pd.testing.assert_frame_equal(f_full.loc[mask], f_short.loc[mask])


# -- shape, bounds, degradation -----------------------------------------------------------


def test_all_features_present_bounded_and_finite():
    ctx = {key: _daily_ctx() for key in ("spx", "gold", "us10y", "usdjpy", "eurusd")}
    bars = _hourly_bars()
    market = _hourly_bars()
    market["close"] *= 1.5
    feats = build_context_features(bars, timeframe="1h", ctx_frames=ctx, market_bars=market)
    assert list(feats.columns) == list(CONTEXT_FEATURE_NAMES)
    assert len(feats) == len(bars)
    assert np.isfinite(feats.to_numpy()).all()
    assert (feats.abs() <= 1.0 + 1e-9).all().all()


def test_missing_context_degrades_to_neutral():
    bars = _hourly_bars(n=200)
    feats = build_context_features(bars, timeframe="1h", ctx_frames={})
    macro_cols = [c for c in CONTEXT_FEATURE_NAMES if not c.startswith("ctx_btc")]
    macro_cols.remove("ctx_rel_btc_mom")
    assert (feats[macro_cols] == 0.0).all().all()
    assert (feats["ctx_btc_mom"] == 0.0).all()


def test_market_is_self_gives_neutral_relative_features():
    bars = _hourly_bars(n=300)
    feats = build_context_features(bars, timeframe="1h", market_is_self=True)
    assert (feats["ctx_rel_btc_mom"] == 0.0).all()
    assert (feats["ctx_btc_corr"] == 1.0).all()
    assert feats["ctx_btc_mom"].abs().max() > 0.0  # own momentum still informative


def test_relative_strength_sign():
    # Target rallies while BTC stays flat -> rel_btc_mom must end positive.
    n = 600
    ts = T0 + np.arange(n) * HOUR_MS
    target = _ohlcv(ts, 100.0 * (1.002 ** np.arange(n)))
    market = _ohlcv(ts, np.full(n, 100.0) * (1.0 + 0.0001 * np.sin(np.arange(n) / 3.0)))
    feats = build_context_features(target, timeframe="1h", market_bars=market)
    assert feats["ctx_rel_btc_mom"].iloc[-1] > 0.5


# -- the augmenter (store-level integration) ----------------------------------------------


def test_augmenter_stamps_columns_and_resolves_market(paper_settings):
    store = OHLCVStore(paper_settings.data_dir / "ohlcv")
    # An altcoin series alongside the fixture's BTC/JPY so market resolution kicks in.
    n = 200
    ts = 1_600_000_000_000 + np.arange(n) * HOUR_MS
    store.upsert("bitbank", "XRP/JPY", "1h", _ohlcv(ts, 50.0 + np.arange(n) * 0.01))

    from engine.core.config import load_universe

    universe = load_universe(paper_settings.universe_path)
    augmenter = ContextAugmenter.from_settings(paper_settings, universe)

    bars = store.read("bitbank", "XRP/JPY", "1h")
    out = augmenter.augment(bars, source="bitbank", symbol="XRP/JPY", timeframe="1h")
    for col in CONTEXT_FEATURE_NAMES:
        assert col in out.columns
    assert len(out) == len(bars)
    # BTC/JPY exists in the fixture store -> market features come from it (corr defined).
    assert out["ctx_btc_corr"].abs().max() <= 1.0 + 1e-9
    # No macro context synced in tmp store -> macro features neutral, never NaN.
    assert np.isfinite(out[list(CONTEXT_FEATURE_NAMES)].to_numpy()).all()
    assert (out["ctx_spx_mom"] == 0.0).all()


def test_augmenter_empty_bars_passthrough(paper_settings):
    from engine.core.config import load_universe

    universe = load_universe(paper_settings.universe_path)
    augmenter = ContextAugmenter.from_settings(paper_settings, universe)
    empty = pd.DataFrame({c: [] for c in ("ts", "open", "high", "low", "close", "volume")})
    out = augmenter.augment(empty, source="bitbank", symbol="BTC/JPY", timeframe="1h")
    assert len(out) == 0


# -- GP terminal integration --------------------------------------------------------------


def test_gp_features_include_context_and_zero_fill():
    from engine.gp.features import FEATURE_NAMES, build_features

    bars = _hourly_bars(n=100)
    feats = build_features(bars)  # plain bars: ctx terminals must exist as neutral zeros
    for name in CONTEXT_FEATURE_NAMES:
        assert name in FEATURE_NAMES
        assert name in feats
        assert (feats[name] == 0.0).all()

    stamped = bars.copy()
    stamped["ctx_risk_on"] = 0.5
    feats = build_features(stamped)
    assert (feats["ctx_risk_on"] == 0.5).all()


def test_old_gp_programs_still_evaluate():
    # A hall-of-fame program serialized before the context layer references only base
    # terminals; it must deserialize and evaluate unchanged.
    from engine.gp.features import build_features
    from engine.gp.program import deserialize, evaluate, serialize

    legacy = ["op", "add", [["feat", "ret"], ["const", 1.5]]]
    node = deserialize(legacy)
    bars = _hourly_bars(n=50)
    signal = evaluate(node, build_features(bars))
    assert len(signal) == len(bars)
    assert serialize(node) == legacy
