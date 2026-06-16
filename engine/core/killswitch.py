"""The commit-based kill switch.

A single committed file, ``state/KILL_SWITCH``, is the emergency stop. If it exists,
every executor run must flatten and halt *before* placing any new orders. The user can
engage it from any device by committing the file — no infrastructure required.
"""

from __future__ import annotations

from pathlib import Path

KILL_SWITCH_FILENAME = "KILL_SWITCH"


def kill_switch_path(state_dir: Path) -> Path:
    return Path(state_dir) / KILL_SWITCH_FILENAME


def is_engaged(state_dir: Path) -> bool:
    """True if the kill switch file is present."""
    return kill_switch_path(state_dir).exists()


def engage(state_dir: Path, reason: str = "") -> Path:
    """Create the kill-switch file (used by automated halts; commit it to persist)."""
    path = kill_switch_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"engaged\nreason: {reason}\n", encoding="utf-8")
    return path


def disengage(state_dir: Path) -> None:
    """Remove the kill-switch file if present."""
    kill_switch_path(state_dir).unlink(missing_ok=True)
