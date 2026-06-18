"""Portfolio construction: how capital is split across champions, and regime-aware scaling.

Replaces the M4/M5 equal-weight placeholder with confidence-weighted risk parity:
allocate more to lower-volatility, statistically-confident, *uncorrelated* champions, with
hard caps (per strategy, and a tight cap on the GP family), then scale exposure down in
turbulent regimes. All deterministic — no randomness in the live path.
"""

from engine.portfolio.allocate import AllocationConfig, AllocInput, allocate
from engine.portfolio.regime import RegimeConfig, exposure_scale

__all__ = ["AllocationConfig", "AllocInput", "allocate", "RegimeConfig", "exposure_scale"]
