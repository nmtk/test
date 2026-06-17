import numpy as np
import pandas as pd

from engine.core.config import Settings, load_settings
from engine.data.store import OHLCVStore
from engine.orchestration.champions import load_champions
from engine.orchestration.population import Population, StrategyRecord
from engine.orchestration.promotion import (
    PromotionConfig,
    allocate_champions,
    decay_detected,
    incubation_passed,
    promote_step,
)

CFG = PromotionConfig(min_incubation_bars=60, min_decay_bars=30, max_champions=4)


# -- unit: incubation -------------------------------------------------------------------


def test_incubation_passes_when_forward_within_band():
    fwd = pd.Series(np.random.default_rng(0).normal(0.01, 0.01, 120))  # positive forward
    ref = {"sharpe_per_bar": 0.05, "n": 300, "skew": 0.0, "kurtosis": 3.0}
    assert incubation_passed(fwd, ref, CFG) is True


def test_incubation_fails_too_few_bars():
    fwd = pd.Series(np.random.default_rng(0).normal(0.01, 0.01, 10))
    assert incubation_passed(fwd, {"sharpe_per_bar": 0.0, "n": 300}, CFG) is False


def test_incubation_fails_when_forward_negative():
    fwd = pd.Series(np.random.default_rng(0).normal(-0.01, 0.01, 120))
    assert incubation_passed(fwd, {"sharpe_per_bar": 0.05, "n": 300}, CFG) is False


# -- unit: decay ------------------------------------------------------------------------


def test_no_decay_when_forward_matches_backtest():
    rng = np.random.default_rng(0)
    ref = {
        "sharpe_per_bar": 0.1,
        "n": 300,
        "skew": 0.0,
        "kurtosis": 3.0,
        "mean_per_bar": 0.001,
        "std_per_bar": 0.01,
        "max_drawdown": -0.2,
    }
    fwd = pd.Series(rng.normal(0.001, 0.01, 150))
    assert decay_detected(fwd, ref, CFG) == []


def test_decay_when_forward_collapses():
    rng = np.random.default_rng(0)
    ref = {
        "sharpe_per_bar": 0.1,
        "n": 300,
        "skew": 0.0,
        "kurtosis": 3.0,
        "mean_per_bar": 0.001,
        "std_per_bar": 0.01,
        "max_drawdown": -0.05,
    }
    fwd = pd.Series(rng.normal(-0.02, 0.01, 150))
    reasons = decay_detected(fwd, ref, CFG)
    # a uniformly bad forward is caught by the Sharpe band + drawdown (Page-Hinkley fires on
    # a mid-stream *shift*, which this constant-mean series does not have)
    assert "below_sharpe_band" in reasons
    assert "drawdown" in reasons


# -- unit: allocation -------------------------------------------------------------------


def test_allocate_champions_top_n_equal_weight(tmp_path):
    pop = Population(tmp_path / "pop.db")
    for i, dsr in enumerate([0.99, 0.80, 0.60]):
        pop.upsert(
            StrategyRecord(
                family=f"tsmom:S{i}/JPY:1h",
                strategy="tsmom",
                symbol=f"S{i}/JPY",
                source="bitbank",
                timeframe="1h",
                params={"lookback": 50},
                status="champion",
                metrics={"dsr": dsr, "cpcv_median": dsr},
            )
        )
    champs = allocate_champions(pop, max_champions=2, gross=1.0)
    assert len(champs.allocations) == 2
    assert {a.symbol for a in champs.allocations} == {"S0/JPY", "S1/JPY"}
    assert all(abs(a.weight - 0.5) < 1e-9 for a in champs.allocations)
    pop.close()


# -- FSM end-to-end ---------------------------------------------------------------------


def _settings(tmp_path) -> Settings:
    base = load_settings()
    return Settings(
        repo_root=base.repo_root,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        config_dir=base.config_dir,
        live_enabled=False,
        alpaca_paper=True,
        bitbank=base.bitbank,
        alpaca=base.alpaca,
    )


def _seed_bars(settings, kind="trend", n=400, seed=0):
    rng = np.random.default_rng(seed)
    # trend: upward drift (tsmom earns) ; choppy: no drift + high noise (tsmom whipsaws)
    steps = rng.normal(0.002, 0.01, n) if kind == "trend" else rng.normal(0.0, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    ts = np.arange(n) * 3_600_000 + 1_600_000_000_000
    df = pd.DataFrame(
        {"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 1.0}
    )
    OHLCVStore(settings.data_dir / "ohlcv").upsert("bitbank", "BTC/JPY", "1h", df)
    return int(ts[50])  # early start so the forward window is large


def _ref(**over):
    base = {
        "dsr": 0.99,
        "cpcv_median": 1.0,
        "sharpe_per_bar": 0.0,
        "n": 200,
        "skew": 0.0,
        "kurtosis": 3.0,
        "mean_per_bar": 0.0,
        "std_per_bar": 0.01,
        "max_drawdown": -0.5,
        "ppy": 365.0,
    }
    base.update(over)
    return base


def test_promote_step_incubating_to_champion(tmp_path):
    settings = _settings(tmp_path)
    start = _seed_bars(settings)
    pop = Population(settings.state_dir / "population.db")
    pop.upsert(
        StrategyRecord(
            family="tsmom:BTC/JPY:1h",
            strategy="tsmom",
            symbol="BTC/JPY",
            source="bitbank",
            timeframe="1h",
            params={"lookback": 20},
            status="incubating",
            metrics=_ref(),
            incubation_start_ts=start,
        )
    )
    pop.close()

    summary = promote_step(settings, cfg=CFG)
    assert summary.n_promoted == 1
    assert summary.n_champions == 1

    pop = Population(settings.state_dir / "population.db")
    row = pop.get("tsmom:BTC/JPY:1h")
    assert row["status"] == "champion"
    assert row["champion_start_ts"] is not None
    pop.close()
    assert len(load_champions(settings.state_dir / "champions.json").allocations) == 1


def test_promote_step_retires_decayed_champion(tmp_path):
    settings = _settings(tmp_path)
    start = _seed_bars(settings, kind="choppy")  # tsmom whipsaws -> low forward Sharpe
    pop = Population(settings.state_dir / "population.db")
    # An absurd backtest Sharpe the forward can't possibly match -> decay via below-band.
    pop.upsert(
        StrategyRecord(
            family="tsmom:BTC/JPY:1h",
            strategy="tsmom",
            symbol="BTC/JPY",
            source="bitbank",
            timeframe="1h",
            params={"lookback": 20},
            status="champion",
            metrics=_ref(sharpe_per_bar=5.0, max_drawdown=-0.001),
            champion_start_ts=start,
        )
    )
    pop.close()

    summary = promote_step(settings, cfg=CFG)
    assert summary.n_retired == 1
    assert summary.n_champions == 0

    pop = Population(settings.state_dir / "population.db")
    assert pop.get("tsmom:BTC/JPY:1h")["status"] == "retired"
    pop.close()
