import pandas as pd
import pytest

from engine.backtest import run_backtest_panel
from engine.strategies import get_xs_strategy
from engine.strategies.cross_sectional import CrossSectionalMomentum


def _panel():
    return pd.DataFrame(
        {
            "A": [100, 110, 121, 133.1, 146.41],  # winner
            "B": [100, 90, 81, 72.9, 65.61],  # loser
        },
        dtype=float,
    )


def test_registered():
    assert get_xs_strategy("xs_momentum") is CrossSectionalMomentum


def test_longs_winner_shorts_loser():
    closes = _panel()
    w = CrossSectionalMomentum(lookback=2, top_frac=0.5).weights_panel(closes)
    # first two rows have no momentum (lookback shift) -> flat
    assert (w.iloc[:2] == 0.0).all().all()
    # thereafter: long A (+0.5), short B (-0.5)
    assert w.iloc[-1]["A"] == pytest.approx(0.5)
    assert w.iloc[-1]["B"] == pytest.approx(-0.5)


def test_gross_exposure_capped():
    w = CrossSectionalMomentum(lookback=2, top_frac=0.5).weights_panel(_panel())
    assert (w.abs().sum(axis=1) <= 1.0 + 1e-9).all()


def test_dollar_neutral_rows():
    w = CrossSectionalMomentum(lookback=2, top_frac=0.5).weights_panel(_panel())
    active = w[w.abs().sum(axis=1) > 0]
    assert active.sum(axis=1).abs().max() == pytest.approx(0.0, abs=1e-12)


def test_profitable_on_divergence():
    closes = _panel()
    w = CrossSectionalMomentum(lookback=2, top_frac=0.5).weights_panel(closes)
    r = run_backtest_panel(closes, w)
    assert r.equity.iloc[-1] > 1.0


def test_flat_when_too_few_names():
    closes = pd.DataFrame({"A": [1, 2, 3, 4, 5]}, dtype=float)
    w = CrossSectionalMomentum(lookback=2, top_frac=0.5).weights_panel(closes)
    assert (w == 0.0).all().all()  # need >= 2 names to rank
