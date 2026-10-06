import numpy as np
import pandas as pd
import pytest

from engine.core.config import Settings, load_settings
from engine.data.store import OHLCVStore
from engine.orchestration.champions import Allocation, Champions, save_champions


@pytest.fixture
def paper_settings(tmp_path):
    """A Settings pointing at tmp data/state (real config), seeded with synthetic BTC/JPY
    bars and a one-allocation champions.json — enough to drive the paper executor E2E."""
    base = load_settings()
    settings = Settings(
        repo_root=base.repo_root,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        config_dir=base.config_dir,
        live_enabled=False,
        alpaca_paper=True,
        bitbank=base.bitbank,
        alpaca=base.alpaca,
    )
    n = 200
    close = 100.0 * (1.01 ** np.arange(n))  # steady uptrend -> tsmom goes long
    ts = np.arange(n) * 3_600_000 + 1_600_000_000_000
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
        }
    )
    OHLCVStore(settings.data_dir / "ohlcv").upsert("bitbank", "BTC/JPY", "1h", df)
    save_champions(
        settings.state_dir / "champions.json",
        Champions(
            allocations=[Allocation("tsmom", "BTC/JPY", "bitbank", "1h", 1.0, {"lookback": 20})],
            updated="test",
        ),
    )
    return settings


def _synthetic_ticks(
    hours: float = 30.0,
    seed: int = 0,
    start_ms: int = 1_704_067_200_000,  # 2024-01-01T00:00Z
    jump_rate: float = 0.002,
    wide_rate: float = 0.002,
) -> pd.DataFrame:
    """USD/JPY-like ticks in integer feed points (1 point = 0.001 = 0.1 pip).

    A random walk of 1-3 point steps every 0.2-4s, with occasional multi-pip jumps and
    occasional spread blow-outs, so intrabar tests see gaps, wide quotes and minutes that
    touch both a take-profit and a stop.
    """
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / 2.1)
    ts = start_ms + np.cumsum(rng.integers(200, 4000, n)).astype("int64")
    step = rng.integers(-3, 4, n)
    jumps = rng.random(n) < jump_rate
    step[jumps] = rng.integers(-60, 61, int(jumps.sum()))
    bid = 150_000 + np.cumsum(step)
    spread = rng.integers(2, 7, n)
    wide = rng.random(n) < wide_rate
    spread[wide] = rng.integers(20, 60, int(wide.sum()))
    return pd.DataFrame({"ts": ts, "bid": bid / 1000.0, "ask": (bid + spread) / 1000.0})


@pytest.fixture
def make_ticks():
    return _synthetic_ticks
