from engine.orchestration.orchestrator import run_step
from engine.reporting import generate_report


def test_report_after_step(paper_settings):
    run_step(paper_settings, mode="paper")
    path = generate_report(paper_settings, mode="paper")
    assert path.exists()
    text = path.read_text()
    assert "# trade — paper report" in text
    assert "## Champions" in text
    assert "## Equity" in text
    assert "## Pipeline" in text
    assert "BTC/JPY" in text
    # a dated copy is written too
    dated = list((paper_settings.state_dir / "reports").glob("*-paper.md"))
    assert dated


def test_report_without_state(paper_settings):
    # no step run yet -> still produces a valid report (flat / no history)
    path = generate_report(paper_settings, mode="paper")
    text = path.read_text()
    assert "## Positions" in text
    assert "_flat_" in text
    assert "## Pipeline" in text
