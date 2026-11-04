"""The forward window is a *duration*, not a bar count.

Regression suite for the second defect the shadow ledger exposed on its first night. The
promotion test required ``min_incubation_bars`` forward bars with no reference to how long
a bar is, so on fine timeframes the window collapsed:

    60 bars of 5m data = 5 hours.

The standard error of a Sharpe estimated from 144 observations is about 1/sqrt(144) =
0.083 per bar. The first batch of real verdicts measured 0.055 per bar on exactly that
window — indistinguishable from zero — and then annualised it by sqrt(105120) to print
**+15.1**, which reads like a spectacular result and is noise. Two of the twelve verdicts
were recorded as "would have cleared the promotion test" on that basis.

The same code path promotes strategies to live capital, so this was not only a reporting
problem: a 5m strategy could have become a champion on five hours of forward data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.orchestration.promotion import (
    PromotionConfig,
    incubation_passed,
    required_forward_bars,
)


def _ref(sharpe_per_bar: float = 0.05, n: int = 1200) -> dict:
    return {
        "sharpe_per_bar": sharpe_per_bar,
        "n": n,
        "skew": 0.0,
        "kurtosis": 3.0,
        "ppy": 365.0,
    }


@pytest.mark.parametrize(
    "timeframe,expected",
    [
        ("1d", 60),  # unchanged: 60 bars is what 60 days always meant here
        ("4h", 360),
        ("1h", 1_440),
        ("5m", 17_280),
        ("1m", 86_400),
    ],
)
def test_sixty_days_is_sixty_days_on_every_timeframe(timeframe, expected):
    assert required_forward_bars(timeframe, PromotionConfig()) == expected


def test_the_bar_count_is_a_floor_not_a_substitute():
    """A coarse timeframe must still clear min_incubation_bars."""
    cfg = PromotionConfig(min_incubation_bars=60, min_incubation_days=10.0)
    assert required_forward_bars("1d", cfg) == 60, "10 days of daily bars is only 10 bars"
    assert required_forward_bars("1h", cfg) == 240


def test_five_hours_of_five_minute_bars_cannot_promote_anything():
    """The exact shape of the defect: a strong-looking Sharpe on a window that proves nothing."""
    # 144 bars with per-bar Sharpe ~0.055 — the real gp:USD/JPY:5m verdict.
    rng = np.random.default_rng(0)
    forward = pd.Series(rng.normal(0.000_055, 0.001, 144))
    cfg = PromotionConfig()
    assert not incubation_passed(forward, _ref(), cfg, "5m"), "5 hours is not evidence"
    # ...and the bar-count-only form, which is what shipped, would have let it through.
    assert len(forward) >= cfg.min_incubation_bars


def test_a_long_enough_window_is_still_judged_on_its_merits():
    rng = np.random.default_rng(1)
    strong = pd.Series(rng.normal(0.002, 0.01, 20_000))
    weak = pd.Series(rng.normal(-0.002, 0.01, 20_000))
    cfg = PromotionConfig()
    assert incubation_passed(strong, _ref(), cfg, "5m")
    assert not incubation_passed(weak, _ref(), cfg, "5m")


def test_omitting_the_timeframe_falls_back_to_the_bar_count():
    """Kept for callers that predate the duration rule; production always passes one."""
    forward = pd.Series(np.random.default_rng(2).normal(0.002, 0.01, 80))
    assert incubation_passed(forward, _ref(), PromotionConfig(), None)


def test_an_unparseable_timeframe_does_not_crash_the_fsm():
    with pytest.raises(ValueError):
        required_forward_bars("not-a-timeframe", PromotionConfig())
