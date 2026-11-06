"""Every workflow that commits ``state/`` must be serialized against the others.

Regression suite for two discarded nightly research runs (2026-10-31 and 2026-11-06).
Three workflows run ``trade promote`` and commit ``state/``: executor-paper on an hourly
cron, executor-live, and research-nightly. They each had their own concurrency group, so
they could and did overlap. When they do, the second push is rejected and the rebase hits
the files they both wrote:

  * ``state/champions.json`` — fixed at the source as well (see test_champions.py), but
  * ``state/population.db`` is a **binary SQLite file**, and no rebase can merge one.

The nightly's push guard refuses to guess at a conflict and exits, so the whole
twenty-minute research run — search, GP evolution, the population update — is thrown away.
A shared concurrency group with ``cancel-in-progress: false`` makes the overlap impossible
by queueing instead.

This is checked as a test rather than left to the comments because the failure is silent:
the research simply never lands, and the only trace is a red workflow among hundreds of
green ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
SHARED_GROUP = "state-writers"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _all_workflows() -> dict[str, dict]:
    return {p.name: yaml.safe_load(p.read_text()) for p in sorted(WORKFLOWS.glob("*.yml"))}


def _commits_state(workflow: dict) -> bool:
    for job in (workflow.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            if "state/" in str(step.get("run", "")):
                return True
    return False


def test_the_state_writers_are_exactly_the_ones_we_think():
    """If a new workflow starts committing state/, this test is how we find out."""
    writers = {name for name, wf in _all_workflows().items() if _commits_state(wf)}
    assert writers == {
        "executor-live.yml",
        "executor-paper.yml",
        "research-nightly.yml",
    }, f"a workflow began committing state/ without joining the shared group: {writers}"


@pytest.mark.parametrize(
    "name", ["research-nightly.yml", "executor-paper.yml", "executor-live.yml"]
)
def test_every_state_writer_shares_one_concurrency_group(name):
    concurrency = _load(name).get("concurrency")
    assert concurrency, f"{name} has no concurrency block"
    assert concurrency["group"] == SHARED_GROUP, (
        f"{name} can overlap the other state writers; an overlap discards "
        "the nightly research run on a binary population.db conflict"
    )


@pytest.mark.parametrize(
    "name", ["research-nightly.yml", "executor-paper.yml", "executor-live.yml"]
)
def test_a_queued_state_writer_waits_rather_than_cancelling_the_running_one(name):
    """``cancel-in-progress: true`` would kill the nightly mid-run instead of queueing."""
    assert _load(name)["concurrency"]["cancel-in-progress"] is False


def test_the_data_syncs_stay_out_of_the_shared_group():
    """They only write data/, so serializing them behind a 20-minute research run would
    stall the feeds for nothing."""
    for name in ("data-sync.yml", "data-sync-fx.yml"):
        assert _load(name)["concurrency"]["group"] != SHARED_GROUP


def test_the_nightly_still_runs_the_shadow_step():
    """The step that measures the gate's false-negative rate is easy to lose in an edit."""
    runs = [
        str(step.get("run", ""))
        for job in _load("research-nightly.yml")["jobs"].values()
        for step in job["steps"]
    ]
    assert any("trade shadow" in r for r in runs)
    assert any("trade research" in r for r in runs)
    assert any("trade promote" in r for r in runs)
