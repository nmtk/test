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
