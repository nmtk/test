from engine.core.types import AssetClass, Instrument, Venue
from engine.orchestration.risk import RiskLimits, apply_limits, can_short

BTC = Instrument("BTC/JPY", AssetClass.CRYPTO, Venue.BITBANK)
AAPL = Instrument("AAPL", AssetClass.EQUITY, Venue.ALPACA)
INSTR = {"BTC/JPY": BTC, "AAPL": AAPL}


def test_can_short():
    assert can_short(AAPL) is True
    assert can_short(BTC) is False  # bitbank spot is long-only


def test_long_only_clamps_crypto_shorts():
    out = apply_limits({"BTC/JPY": -0.5}, INSTR, RiskLimits())
    assert out["BTC/JPY"] == 0.0


def test_equity_short_allowed():
    out = apply_limits({"AAPL": -0.4}, INSTR, RiskLimits(max_position_weight=1.0))
    assert out["AAPL"] == -0.4


def test_per_position_cap():
    out = apply_limits({"AAPL": 5.0}, INSTR, RiskLimits(max_position_weight=0.3))
    assert out["AAPL"] == 0.3


def test_gross_exposure_scaling():
    out = apply_limits(
        {"BTC/JPY": 0.8, "AAPL": 0.8},
        INSTR,
        RiskLimits(max_gross_exposure=1.0, max_position_weight=1.0),
    )
    assert abs(out["BTC/JPY"]) + abs(out["AAPL"]) == 1.0
    assert out["BTC/JPY"] == 0.5
