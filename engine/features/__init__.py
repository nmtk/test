"""Cross-asset context features: where money is flowing *outside* the target's own chart.

The engine's original hypothesis space saw only the target instrument's OHLCV. This
package widens it: macro series (equities, gold, rates, FX) and the crypto market factor
(BTC) are joined onto the target's bars as causal ``ctx_*`` columns, so both the GP
generator and human-designed archetypes can condition on *flows between assets* —
risk-on/risk-off, dollar liquidity, BTC-to-altcoin rotation — not just price patterns.
"""

from engine.features.context import (
    CONTEXT_FEATURE_NAMES,
    CONTEXT_KEYS,
    ContextAugmenter,
    build_context_features,
)

__all__ = [
    "CONTEXT_FEATURE_NAMES",
    "CONTEXT_KEYS",
    "ContextAugmenter",
    "build_context_features",
]
