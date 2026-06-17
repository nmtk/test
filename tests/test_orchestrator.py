from engine.core.killswitch import engage
from engine.orchestration import state as state_io
from engine.orchestration.orchestrator import run_step


def test_first_step_opens_position(paper_settings):
    r = run_step(paper_settings, mode="paper")
    assert r.note == "ok"
    assert r.n_orders == 1
    assert r.positions["BTC/JPY"] > 0.0  # tsmom long in an uptrend
    assert "JPY" in r.equity
    # state was persisted
    assert state_io.account_path(paper_settings.state_dir, "paper").exists()
    assert state_io.equity_path(paper_settings.state_dir, "paper").exists()


def test_second_step_is_idempotent(paper_settings):
    run_step(paper_settings, mode="paper")
    r2 = run_step(paper_settings, mode="paper")
    # same bar, position already on target -> no churn thanks to the no-trade band
    assert r2.n_orders == 0
    assert r2.positions["BTC/JPY"] > 0.0


def test_kill_switch_flattens(paper_settings):
    run_step(paper_settings, mode="paper")
    engage(paper_settings.state_dir, reason="test")
    r = run_step(paper_settings, mode="paper")
    assert r.killed is True
    assert r.n_orders == 1  # the flattening sell
    assert r.positions == {}  # flat


def test_no_champions_is_safe(tmp_path, paper_settings):
    # point at an empty state dir (no champions.json) -> no data, no crash
    from engine.core.config import Settings

    empty = Settings(
        repo_root=paper_settings.repo_root,
        data_dir=paper_settings.data_dir,
        state_dir=tmp_path / "empty_state",
        config_dir=paper_settings.config_dir,
        live_enabled=False,
        alpaca_paper=True,
        bitbank=paper_settings.bitbank,
        alpaca=paper_settings.alpaca,
    )
    r = run_step(empty, mode="paper")
    assert r.n_orders == 0
    assert r.note == "no data"
