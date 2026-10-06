"""Seed strategy archetypes and the strategy registry.

Each strategy maps a bar history to a *target weight* series in ``[-1, 1]`` (the desired
signed exposure as a fraction of equity), computed causally. The same method serves both
research (use the whole series) and live execution (take the last value). Importing this
package registers all built-in archetypes.
"""

from engine.strategies.base import (
    CrossSectionalStrategy,
    Strategy,
    all_strategies,
    all_xs_strategies,
    get_strategy,
    get_xs_strategy,
    register,
    register_xs,
)
from engine.strategies.breakout import DonchianBreakout
from engine.strategies.cross_sectional import CrossSectionalMomentum
from engine.strategies.ensemble import EnsembleStrategy
from engine.strategies.macro_flow import MacroFlowTrend
from engine.strategies.mean_reversion import ZScoreReversion
from engine.strategies.momentum import MovingAverageCross, TimeSeriesMomentum
from engine.strategies.trade_balance import TradeBalanceUsdJpy
from engine.strategies.wemof import WemofReversal
from engine.strategies.wemof_tick import WemofTick

__all__ = [
    "Strategy",
    "CrossSectionalStrategy",
    "register",
    "register_xs",
    "get_strategy",
    "get_xs_strategy",
    "all_strategies",
    "all_xs_strategies",
    "TimeSeriesMomentum",
    "MovingAverageCross",
    "DonchianBreakout",
    "ZScoreReversion",
    "CrossSectionalMomentum",
    "EnsembleStrategy",
    "MacroFlowTrend",
    "WemofReversal",
    "WemofTick",
    "TradeBalanceUsdJpy",
]
