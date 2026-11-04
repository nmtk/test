"""Shadow tracking of gate-rejected candidates.

The gate is a classifier whose false-negative rate has never been measured. These tests
pin down the three properties that make the measurement worth anything:

  * it measures the *rejects*, and leaves the gate-passers to the promotion FSM;
  * it judges them with the identical test a promoted strategy faces, so the two groups
    are comparable at all;
  * a track is immutable — it keeps measuring the params it stamped, even though the
    nightly search overwrites the population row underneath it. Measuring a row that
    changes every night is how the incubation clock came to be stuck at one bar
    (tests/test_incubation_clock.py); the same mistake here would produce a confident
    number about nothing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from engine.core.config import load_settings
from engine.data.store import OHLCVStore
from engine.orchestration.population import Population, StrategyRecord
from engine.orchestration.promotion import PromotionConfig
from engine.orchestration.shadow import (
    MAX_VERDICTS_PER_FAMILY,
    WORTH_TRADING_SHARPE,
    ShadowLedger,
    params_hash,
    shadow_step,
)

DAY_MS = 86_400_000
FAMILY = "tsmom:BTC/JPY:1d"


def _settings_at(tmp_path):
    base = load_settings()
    return dataclasses.replace(base, state_dir=tmp_path / "state", data_dir=tmp_path / "data")


def _write_bars(
    tmp_path, n: int, *, trend: float = 0.0, noise: float = 0.0, whipsaw: bool = False
) -> int:
    """A deterministic series.

    ``noise`` matters more than it looks: a perfectly smooth series has zero return
    variance, so ``mean / std`` blows up to a meaningless Sharpe of 1e16 and every band
    test passes for the wrong reason. Any test that asserts something about a *band* needs
    a series with real dispersion.
    """
    index = np.arange(n)
    if whipsaw:
        close = 1_000_000.0 * (1.0 + 0.02 * ((index // 3) % 2))
    else:
        steps = np.full(n, 1.0 + trend)
        if noise:
            steps = steps + np.random.default_rng(0).normal(0.0, noise, n)
        close = 1_000_000.0 * np.cumprod(steps)
    frame = pd.DataFrame(
        {
            "ts": index.astype("int64") * DAY_MS + 1_700_000_000_000,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )
    store = OHLCVStore(tmp_path / "data" / "ohlcv")
    path = store.path("bitbank", "BTC/JPY", "1d")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return int(frame["ts"].iloc[0])


def _gate_metrics(*, passed: bool, sharpe_per_bar: float = 0.05) -> dict:
    return {
        "ppy": 365.0,
        "n": 1200,
        "sharpe_per_bar": sharpe_per_bar,
        "sharpe_annual": sharpe_per_bar * 365.0**0.5,
        "skew": 0.0,
        "kurtosis": 3.0,
        "max_drawdown": -0.05,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "dsr": 0.4,
        "pbo": 0.6,
        "spa_p": 0.3,
        "passed": passed,
        "research_only": False,
    }


def _seed(
    settings,
    *,
    status="candidate",
    passed=False,
    params=None,
    strategy="tsmom",
    family=FAMILY,
    symbol="BTC/JPY",
    sharpe_per_bar=0.05,
):
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    population = Population(settings.state_dir / "population.db")
    population.upsert(
        StrategyRecord(
            family=family,
            strategy=strategy,
            symbol=symbol,
            source="bitbank",
            timeframe="1d",
            params=params if params is not None else {"lookback": 5},
            status=status,
            metrics=_gate_metrics(passed=passed, sharpe_per_bar=sharpe_per_bar),
        )
    )
    population.close()


def _ledger(settings) -> ShadowLedger:
    return ShadowLedger(settings.state_dir / "shadow")


# -- who gets tracked ------------------------------------------------------------------


def test_a_rejected_candidate_is_tracked_from_the_last_bar(tmp_path):
    settings = _settings_at(tmp_path)
    first_ts = _write_bars(tmp_path, 400, trend=0.001)
    _seed(settings)
    summary = shadow_step(settings)
    assert (summary.n_opened, summary.n_tracking) == (1, 1)

    ledger = _ledger(settings)
    (track,) = list(ledger.tracks.values())
    assert track["family"] == FAMILY
    assert track["start_ts"] == first_ts + 399 * DAY_MS, "the window starts at the newest bar"
    assert track["params"] == {"lookback": 5}


def test_a_gate_passer_is_left_to_the_promotion_fsm(tmp_path):
    """Incubating rows already have a forward test; tracking them twice proves nothing."""
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.001)
    _seed(settings, status="incubating", passed=True)
    assert shadow_step(settings).n_opened == 0


def test_a_candidate_row_that_passed_is_not_treated_as_a_reject(tmp_path):
    """A row can be a candidate without having been rejected; only rejects are tracked."""
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.001)
    _seed(settings, status="candidate", passed=True)
    assert shadow_step(settings).n_opened == 0


def test_an_intrabar_strategy_is_skipped_rather_than_mismeasured(tmp_path):
    """wemof_tick's edge is its intrabar pip exits; bar closes cannot see them."""
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.001)
    _seed(
        settings,
        strategy="wemof_tick",
        family="wemof_tick:BTC/JPY:1d",
        params={"tp_pips": 1.0, "sl_pips": 5.0},
    )
    summary = shadow_step(settings)
    assert (summary.n_opened, summary.n_skipped_intrabar) == (0, 1)


def test_a_series_with_no_stored_bars_is_skipped(tmp_path):
    settings = _settings_at(tmp_path)  # no bars written at all
    _seed(settings)
    assert shadow_step(settings).n_opened == 0


def test_a_second_pass_does_not_duplicate_the_track(tmp_path):
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.001)
    _seed(settings)
    shadow_step(settings)
    second = shadow_step(settings)
    assert second.n_opened == 0 and second.n_tracking == 1
    ledger = _ledger(settings)
    assert len(ledger.tracks) == 1


# -- immutability ----------------------------------------------------------------------


def test_the_track_keeps_its_own_params_when_the_row_changes_underneath(tmp_path):
    """The nightly search rewrites the row; the measurement must not follow it."""
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.001)
    _seed(settings, params={"lookback": 5})
    shadow_step(settings)
    ledger = _ledger(settings)
    (before,) = list(ledger.tracks.values())

    _seed(settings, params={"lookback": 180})  # a different draw the next night
    shadow_step(settings)

    ledger = _ledger(settings)
    (after,) = list(ledger.tracks.values())
    assert after["params"] == {"lookback": 5}
    assert after["start_ts"] == before["start_ts"], "nor may the clock restart"
    assert after["params_hash"] == params_hash({"lookback": 5})


# -- the verdict -----------------------------------------------------------------------


def _mature(settings, *, back_days: int) -> None:
    """Age every open track so its forward window already holds data."""
    ledger = _ledger(settings)
    for track in ledger.tracks.values():
        track["start_ts"] -= back_days * DAY_MS
    ledger.save()


def test_a_short_window_is_not_judged_yet(tmp_path):
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.001)
    _seed(settings)
    shadow_step(settings)
    _mature(settings, back_days=10)
    summary = shadow_step(settings, cfg=PromotionConfig(min_incubation_bars=60))
    assert (summary.n_closed, summary.n_tracking) == (0, 1)


def test_a_reject_that_earns_its_keep_is_recorded_as_a_false_negative(tmp_path):
    """A rising series makes the rejected momentum config work: the gate got this one wrong."""
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.01, noise=0.01)
    _seed(settings, sharpe_per_bar=0.05)
    shadow_step(settings)
    _mature(settings, back_days=100)
    summary = shadow_step(settings, cfg=PromotionConfig(min_incubation_bars=60))
    assert (summary.n_closed, summary.n_worth_trading) == (1, 1)

    ledger = _ledger(settings)
    (verdict,) = ledger.verdicts()
    assert ledger.tally() == (1, 1, 0)
    assert verdict["worth_trading"] is True
    assert verdict["fwd_sharpe_annual"] > 0
    assert verdict["fwd_cum_return"] > 0
    assert verdict["n_bars"] >= 60
    # the gate metrics it was rejected on travel with the verdict, so the binding
    # threshold can be identified later
    assert (verdict["gate_dsr"], verdict["gate_pbo"]) == (0.4, 0.6)
    assert verdict["params"] == {"lookback": 5}


def test_a_reject_that_loses_money_confirms_the_gate_was_right(tmp_path):
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, whipsaw=True)  # momentum loses on this series
    _seed(settings)
    shadow_step(settings)
    _mature(settings, back_days=100)
    summary = shadow_step(settings, cfg=PromotionConfig(min_incubation_bars=60))
    assert (summary.n_closed, summary.n_worth_trading) == (1, 0)
    ledger = _ledger(settings)
    (verdict,) = ledger.verdicts()
    assert verdict["worth_trading"] is False
    assert verdict["fwd_sharpe_annual"] < 0


def test_the_relative_test_is_recorded_but_is_not_the_headline(tmp_path):
    """``would_have_passed`` is kept, and this is why it cannot be the measure of the gate.

    Both runs see the identical forward returns. Only the backtest Sharpe they are judged
    against differs — and that flips the relative verdict while the actual forward
    performance is unchanged. A reject's backtest is weak by definition, so its band sits
    near (or below) zero and the test degenerates into "did the window happen to be up".
    """
    cfg = PromotionConfig(min_incubation_bars=60, min_incubation_days=60.0)

    lenient = _settings_at(tmp_path / "lenient")
    _write_bars(tmp_path / "lenient", 400, trend=0.004, noise=0.01)
    _seed(lenient, sharpe_per_bar=0.01)  # a weak backtest => a trivially low bar
    shadow_step(lenient)
    _mature(lenient, back_days=100)
    shadow_step(lenient, cfg=cfg)

    strict = _settings_at(tmp_path / "strict")
    _write_bars(tmp_path / "strict", 400, trend=0.004, noise=0.01)
    _seed(strict, sharpe_per_bar=5.0)  # a band no real forward stretch can reach
    shadow_step(strict)
    _mature(strict, back_days=100)
    shadow_step(strict, cfg=cfg)

    (weak,) = _ledger(lenient).verdicts()
    (strong,) = _ledger(strict).verdicts()
    assert weak["fwd_sharpe_annual"] == strong["fwd_sharpe_annual"], "same forward returns"
    assert weak["would_have_passed"] and not strong["would_have_passed"]
    assert weak["worth_trading"] == strong["worth_trading"], (
        "the absolute verdict must not depend on the backtest it is compared against"
    )


def test_the_family_is_restamped_in_the_same_pass_that_judged_it(tmp_path):
    """Closing and re-opening happen in one pass, so a family is never idle between windows.

    The re-stamp takes the population row's *current* params, which is the whole point:
    the next window measures whatever the search believes today, on bars nothing has seen.
    """
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.01, noise=0.01)
    _seed(settings, params={"lookback": 5})
    shadow_step(settings)
    _mature(settings, back_days=100)

    _seed(settings, params={"lookback": 90})  # tonight's draw, before the judging pass
    summary = shadow_step(settings, cfg=PromotionConfig(min_incubation_bars=60))
    assert (summary.n_closed, summary.n_opened) == (1, 1)

    ledger = _ledger(settings)
    (track,) = list(ledger.tracks.values())
    (verdict,) = ledger.verdicts()
    assert verdict["params"] == {"lookback": 5}, "the verdict is about the old draw"
    assert track["params"] == {"lookback": 90}, "the new window is about tonight's draw"


def test_unusable_params_are_counted_not_crashed_on(tmp_path):
    """A param the strategy no longer accepts must not take the whole nightly step down."""
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.001)
    _seed(settings, params={"lookback": 5})
    shadow_step(settings)
    ledger = _ledger(settings)
    for track in ledger.tracks.values():
        track["params"] = {"not_a_real_param": 1}
    ledger.save()

    summary = shadow_step(settings, cfg=PromotionConfig(min_incubation_bars=60))
    assert summary.n_errors == 1 and "tsmom:BTC/JPY:1d" in summary.errors[0]

    ledger = _ledger(settings)
    (track,) = list(ledger.tracks.values())
    assert ledger.verdicts() == [], "a track that could not be measured yields no verdict"
    # Dropped and re-stamped from the population row in the same pass, so the family
    # heals itself instead of failing the same way every night forever.
    assert track["params"] == {"lookback": 5}


def test_verdict_history_is_capped_per_family(tmp_path):
    """state/ is committed to git, so an append-only table needs a ceiling."""
    settings = _settings_at(tmp_path)
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    ledger = _ledger(settings)
    for i in range(MAX_VERDICTS_PER_FAMILY + 7):
        ledger.append_verdict(
            family=FAMILY,
            strategy="tsmom",
            symbol="BTC/JPY",
            source="bitbank",
            timeframe="1d",
            params={"lookback": i},
            params_hash=params_hash({"lookback": i}),
            start_ts=i * DAY_MS,
            n_bars=60,
            fwd_sharpe_per_bar=0.0,
            fwd_sharpe_annual=0.0,
            fwd_cum_return=0.0,
            fwd_max_drawdown=0.0,
            would_have_passed=0,
            required_lower_band=0.0,
            gate_dsr=0.1,
            gate_pbo=0.9,
            gate_spa=0.9,
            bt_sharpe_annual=0.0,
        )
    rows = ledger.verdicts()
    assert len(rows) == MAX_VERDICTS_PER_FAMILY
    newest = rows[0]["params"]["lookback"]
    assert newest == MAX_VERDICTS_PER_FAMILY + 6, "the cap must drop the oldest, not the newest"


def test_research_only_rejects_are_tracked_too(tmp_path):
    """Shadow testing places no orders, so a non-venue series is perfectly researchable."""
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.001)
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    population = Population(settings.state_dir / "population.db")
    metrics = _gate_metrics(passed=False) | {"research_only": True}
    population.upsert(
        StrategyRecord(
            family=FAMILY,
            strategy="tsmom",
            symbol="BTC/JPY",
            source="bitbank",
            timeframe="1d",
            params={"lookback": 5},
            status="candidate",
            metrics=metrics,
        )
    )
    population.close()
    assert shadow_step(settings).n_opened == 1


def test_a_window_with_no_position_is_recorded_but_kept_out_of_the_rate(tmp_path):
    """Nine of the first twelve real verdicts were this, and they would have deflated it.

    ``session`` on daily bars returns flat by construction (a day holds one bar, so a
    time-of-day window is meaningless). Its forward returns are all zero. That says
    nothing about whether the gate was right to reject it — it is a configuration that
    stood aside — so it must not sit in the denominator of the false-negative rate.
    """
    settings = _settings_at(tmp_path)
    _write_bars(tmp_path, 400, trend=0.004, noise=0.01)
    _seed(settings, strategy="session", family="session:BTC/JPY:1d", params={"start_hour": 7})
    shadow_step(settings)
    _mature(settings, back_days=100)
    summary = shadow_step(settings, cfg=PromotionConfig(min_incubation_bars=60))

    assert (summary.n_closed, summary.n_never_traded, summary.n_worth_trading) == (1, 1, 0)
    ledger = _ledger(settings)
    (verdict,) = ledger.verdicts()
    assert verdict["n_active_bars"] == 0
    assert verdict["worth_trading"] is False
    n_verdicts, n_worth, n_never = ledger.tally()
    assert (n_verdicts, n_worth, n_never) == (1, 0, 1)
    assert n_verdicts - n_never == 0, "the rate has an empty denominator, not a 0% result"


def test_the_absolute_threshold_is_what_decides(tmp_path, monkeypatch):
    """Identical forward returns, two thresholds, two verdicts.

    Engineering a series to land at a chosen Sharpe is not possible with a random draw —
    over 100 bars the realised mean is dominated by noise. So the rule is tested where it
    lives: the same window is judged against a threshold just under and just over its
    measured Sharpe. "Positive" is not the same as "worth trading", and this is the line.
    """
    import engine.orchestration.shadow as shadow_mod

    def verdict_at(threshold: float, path):
        settings = _settings_at(path)
        _write_bars(path, 400, trend=0.004, noise=0.01)
        _seed(settings)
        shadow_step(settings)
        _mature(settings, back_days=100)
        monkeypatch.setattr(shadow_mod, "WORTH_TRADING_SHARPE", threshold)
        shadow_step(settings, cfg=PromotionConfig(min_incubation_bars=60))
        (verdict,) = _ledger(settings).verdicts()
        return verdict

    probe = verdict_at(WORTH_TRADING_SHARPE, tmp_path / "probe")
    measured = probe["fwd_sharpe_annual"]
    assert probe["n_active_bars"] > 0, "it did trade"

    below = verdict_at(measured - 0.5, tmp_path / "below")
    above = verdict_at(measured + 0.5, tmp_path / "above")
    assert below["fwd_sharpe_annual"] == above["fwd_sharpe_annual"] == measured
    assert below["worth_trading"] is True
    assert above["worth_trading"] is False
