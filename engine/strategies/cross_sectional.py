"""Cross-sectional momentum — long the strongest names, short the weakest.

The canonical multi-asset factor: Jegadeesh-Titman (1993) for equities; Liu-Tsyvinski-Wu
(2022) show size/momentum are the dominant crypto factors too. Operates on a basket, so it
implements the panel interface rather than the single-series one.
"""

from __future__ import annotations

import pandas as pd

from engine.strategies.base import CrossSectionalStrategy, FloatParam, IntParam, register_xs


@register_xs
class CrossSectionalMomentum(CrossSectionalStrategy):
    """Rank by trailing ``lookback`` return; long the top fraction, short the bottom.

    Dollar-neutral and capped at gross 1 (longs sum to +0.5, shorts to -0.5), equal-weight
    within each leg. Rows with fewer than two valid names are flat.
    """

    name = "xs_momentum"
    param_space = {"lookback": IntParam(10, 200), "top_frac": FloatParam(0.1, 0.5)}

    def __init__(self, lookback: int = 90, top_frac: float = 0.34) -> None:
        self.lookback = lookback
        self.top_frac = top_frac

    def weights_panel(self, closes: pd.DataFrame) -> pd.DataFrame:
        momentum = closes / closes.shift(self.lookback) - 1.0
        weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

        for ts, row in momentum.iterrows():
            valid = row.dropna()
            n = len(valid)
            if n < 2:
                continue
            k = max(1, int(n * self.top_frac))
            ordered = valid.sort_values()
            shorts = ordered.index[:k]
            longs = ordered.index[-k:]
            weights.loc[ts, longs] = 0.5 / k
            weights.loc[ts, shorts] = -0.5 / k
        return weights
