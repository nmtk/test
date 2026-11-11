"""The forward test must see the same inputs the gate saw.

Research computes weights on *augmented* bars — every ``ctx_*`` and ``mac_*`` column
stamped by ``ContextAugmenter`` — and so do the paper and live executors. The promotion
FSM did not. It read raw bars from the store, so for any strategy that consults a context
feature the column was simply missing, the signal read neutral, and the strategy came out
flat. The forward test was therefore measuring a *different strategy* than the one it was
judging, and ``incubation_passed`` requires ``fwd.sharpe > 0.0``, which a flat return
series can never satisfy.

Measured on the live population on 2026-11-11, with eleven rows incubating: all three
``gp:*`` rows had **zero** exposure across the whole series on raw bars and 14-20% once
augmented, and eight of the remaining rows differed too. They were not slow to promote —
they were unpromotable, and would have been rejected at their January verdict for a reason
that had nothing to do with their edge.

``shadow.py`` shares ``forward_returns``, so every verdict it would have recorded about a
context-reading family was about a strategy that does not exist.

The telling detail is that ``ContextAugmenter.augment``'s own docstring lists who applies
it — "research, the paper executor, and the live executor" — and the promotion FSM is
absent from that list. The omission was written down and still not caught, because nothing
asserted it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.orchestration.promotion import PromotionConfig, forward_returns
from engine.strategies.carry import RateCarry

DAY_MS = 86_400_000
FAMILY = "carry:USD/JPY:1d"


class _Store:
    """Just enough OHLCVStore surface for forward_returns."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars

    def read(self, source, symbol, timeframe):
        return self._bars.copy()


class _Augmenter:
    """Stands in for ContextAugmenter: stamps the one feature `carry` reads."""

    def __init__(self, level: float = 0.8) -> None:
        self.level = level
        self.calls: list[tuple[str, str, str]] = []

    def augment(self, bars, *, source, symbol, timeframe):
        self.calls.append((source, symbol, timeframe))
        return bars.assign(mac_carry_level=self.level)


def _bars(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 150.0 * np.cumprod(1.0 + rng.normal(0.001, 0.004, n))
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * DAY_MS + 1_700_000_000_000,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


def _record() -> dict:
    return {
        "family": FAMILY,
        "strategy": "carry",
        "symbol": "USD/JPY",
        "source": "oanda",
        "timeframe": "1d",
        # direction/threshold chosen so the signal fires whenever the feature exists
        "params": {"direction": 1, "threshold": 0.1, "smooth": 1, "trend_filter": 0},
        "metrics": {"ppy": 252.0, "fee_bps": 0.0, "slippage_bps": 0.0},
    }


def test_carry_is_flat_without_its_feature():
    """The premise: this is what the FSM was measuring."""
    weights = RateCarry(direction=1, threshold=0.1, smooth=1, trend_filter=0).weights(_bars())
    assert (weights == 0.0).all()


def test_the_unaugmented_forward_series_is_all_zeros():
    fwd = forward_returns(_Store(_bars()), _record(), PromotionConfig(), 1_700_000_000_000, None)
    assert fwd is not None and len(fwd) > 0
    assert np.count_nonzero(np.asarray(fwd, dtype=float)) == 0, (
        "a flat forward series can never satisfy incubation_passed's fwd.sharpe > 0.0"
    )


def test_augmenting_makes_it_the_strategy_that_was_validated():
    augmenter = _Augmenter()
    fwd = forward_returns(
        _Store(_bars()), _record(), PromotionConfig(), 1_700_000_000_000, augmenter
    )
    assert np.count_nonzero(np.asarray(fwd, dtype=float)) > 0
    assert augmenter.calls == [("oanda", "USD/JPY", "1d")], (
        "the augmenter must be told which series it is stamping"
    )


def test_a_price_only_strategy_is_unaffected():
    """The fix must not change what it was already measuring correctly."""
    record = _record() | {"strategy": "tsmom", "params": {"lookback": 20}}
    raw = forward_returns(_Store(_bars()), record, PromotionConfig(), 1_700_000_000_000, None)
    aug = forward_returns(
        _Store(_bars()), record, PromotionConfig(), 1_700_000_000_000, _Augmenter()
    )
    np.testing.assert_allclose(np.asarray(raw, dtype=float), np.asarray(aug, dtype=float))


def test_the_augmenter_argument_is_required():
    """Defaulting it is what let the omission live: a caller must decide."""
    with pytest.raises(TypeError):
        forward_returns(_Store(_bars()), _record(), PromotionConfig(), 1_700_000_000_000)


def test_promote_step_supplies_an_augmenter():
    """The FSM is the caller that was wrong; pin that it now passes one.

    Checked by signature rather than by running the FSM, so the assertion survives a
    refactor of promote_step's internals while still failing if the argument is dropped.
    """
    import inspect

    from engine.orchestration import promotion, shadow

    for module in (promotion, shadow):
        source = inspect.getsource(module)
        calls = [line for line in source.splitlines() if "forward_returns(" in line]
        calls = [c for c in calls if "def forward_returns" not in c]
        assert calls, f"{module.__name__} no longer calls forward_returns"
        assert all("augmenter" in c for c in calls), (
            f"{module.__name__} calls forward_returns without an augmenter: {calls}"
        )
