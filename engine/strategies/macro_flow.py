"""Flow-gated trend: trade the target's trend only when cross-asset flows agree.

The hypothesis space the user cares about most: money rotates *between* assets. Crypto
rallies feed on risk-on equity tape, a soft dollar, easing rates, and (for altcoins)
strength relative to BTC; it bleeds when those reverse. This archetype makes that
testable: a plain trend signal on the target is switched on/off by one cross-asset
context gate, and the optuna search + validation gate decide — per symbol and timeframe —
*which* gate (if any) genuinely helps out of sample.

Requires the ``ctx_*`` columns stamped by :class:`engine.features.context.ContextAugmenter`;
without them every gate reads neutral and the strategy stays flat (safe by construction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategies.base import FloatParam, IntParam, Strategy, register

# Gate name -> ctx_ column. All are bounded to ~[-1, 1] (see engine/features/context.py),
# so one threshold scale works for every gate.
GATE_COLUMNS: dict[str, str] = {
    "risk_on": "ctx_risk_on",  # composite: equities up, dollar & gold not bid
    "spx": "ctx_spx_mom",  # equity risk appetite
    "gold": "ctx_gold_mom",  # flight-to-safety bid
    "us10y": "ctx_us10y_mom",  # rates rising / falling
    "dollar": "ctx_dollar_mom",  # global dollar liquidity
    "usdjpy": "ctx_usdjpy_mom",  # the book's own quote currency
    "btc": "ctx_btc_mom",  # money entering/leaving the crypto market as a whole
    "rel_btc": "ctx_rel_btc_mom",  # BTC -> altcoin rotation (relative strength)
}


@register
class MacroFlowTrend(Strategy):
    """``lookback``-bar trend on the target, active only while ``direction * gate > threshold``.

    ``direction=+1`` trades the "flows in when gate is high" hypothesis, ``-1`` the
    inverse (e.g. crypto benefits from a *falling* dollar or *falling* yields). The gate
    controls participation, not side: while on, the position follows the trend long or
    short; while off, flat.
    """

    name = "macro_flow"
    param_space = {
        "lookback": IntParam(10, 120),
        "gate": list(GATE_COLUMNS),
        "direction": [1, -1],
        "threshold": FloatParam(0.0, 0.6),
    }

    def __init__(
        self,
        lookback: int = 40,
        gate: str = "risk_on",
        direction: int = 1,
        threshold: float = 0.0,
    ) -> None:
        if gate not in GATE_COLUMNS:
            raise ValueError(f"unknown gate {gate!r}; expected one of {sorted(GATE_COLUMNS)}")
        self.lookback = lookback
        self.gate = gate
        self.direction = int(direction)
        self.threshold = float(threshold)

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        column = GATE_COLUMNS[self.gate]
        if column not in bars.columns:
            # No context stamped on these bars: stay flat rather than trade an
            # unconditioned trend the validation never approved.
            return pd.Series(0.0, index=bars.index)
        close = bars["close"].astype(float)
        trend = np.sign(close / close.shift(self.lookback) - 1.0).fillna(0.0)
        gate = bars[column].astype(float).fillna(0.0) * self.direction
        active = (gate > self.threshold).astype(float)
        return (trend * active).astype(float)
