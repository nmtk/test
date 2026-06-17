"""Position-level risk limits applied to target weights before sizing.

Targets are signed fractions of book equity. We enforce: no shorting where the venue can't
(bitbank spot is long-only), a per-symbol cap, and a gross-exposure cap (scale the whole
book down if the sum of absolute weights exceeds the limit). The kill switch overrides
everything: if engaged, every target becomes zero (flatten).
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.core.types import AssetClass, Instrument, Venue


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_gross_exposure: float = 1.0
    max_position_weight: float = 1.0


def can_short(instrument: Instrument) -> bool:
    """Spot crypto (bitbank) is long-only; equities (Alpaca margin) may be shorted."""
    if instrument.asset_class is AssetClass.CRYPTO and instrument.venue is Venue.BITBANK:
        return False
    return True


def apply_limits(
    target_weights: dict[str, float],
    instruments: dict[str, Instrument],
    limits: RiskLimits,
) -> dict[str, float]:
    """Clamp per-symbol weights, forbid shorting where disallowed, then cap gross exposure."""
    clamped: dict[str, float] = {}
    for symbol, weight in target_weights.items():
        instrument = instruments.get(symbol)
        w = max(-limits.max_position_weight, min(limits.max_position_weight, weight))
        if instrument is not None and not can_short(instrument):
            w = max(0.0, w)
        clamped[symbol] = w

    gross = sum(abs(w) for w in clamped.values())
    if gross > limits.max_gross_exposure and gross > 0:
        scale = limits.max_gross_exposure / gross
        clamped = {s: w * scale for s, w in clamped.items()}
    return clamped
