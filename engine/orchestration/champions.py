"""The live "brain": champion allocations the executor trades (committed as data).

A champion is a concrete strategy instance (archetype + params) bound to a symbol/timeframe
with a portfolio weight (signed fraction of book equity). This is *data* that evolves as the
nightly research promotes/retires strategies — not code. Until research (M4) populates it,
``seed`` writes a bootstrap set so the paper executor has something to run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Allocation:
    strategy: str
    symbol: str
    source: str
    timeframe: str
    weight: float
    params: dict = field(default_factory=dict)


@dataclass(slots=True)
class Champions:
    allocations: list[Allocation] = field(default_factory=list)
    updated: str = ""

    def to_dict(self) -> dict:
        return {"updated": self.updated, "allocations": [asdict(a) for a in self.allocations]}


def load_champions(path: Path) -> Champions:
    path = Path(path)
    if not path.exists():
        return Champions()
    raw = json.loads(path.read_text())
    allocations = [
        Allocation(
            strategy=a["strategy"],
            symbol=a["symbol"],
            source=a["source"],
            timeframe=a["timeframe"],
            weight=float(a["weight"]),
            params=a.get("params", {}),
        )
        for a in raw.get("allocations", [])
    ]
    return Champions(allocations=allocations, updated=raw.get("updated", ""))


def save_champions(path: Path, champions: Champions) -> None:
    """Write the allocations, but leave the file untouched when they have not changed.

    ``updated`` therefore means "when the allocations last changed", not "when promote
    last ran". That distinction is the whole point: three workflows run ``trade promote``
    (executor-paper hourly, executor-live, research-nightly) and all three commit
    ``state/``. When this rewrote the file unconditionally, each run produced a new
    ``updated`` timestamp on the same line, so an hourly paper step landing inside the
    nightly research window gave a *guaranteed* rebase conflict on champions.json — and
    the nightly's push guard then discarded the whole twenty-minute research run rather
    than guess at a resolution. That happened on 2026-10-31 and again on 2026-11-06,
    silently, while the allocations on both sides were identical (both empty).

    A no-op write also keeps the hourly churn out of git history, since the file is
    committed on every executor run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists() and load_champions(path).allocations == champions.allocations:
            return
    except (OSError, ValueError, KeyError, TypeError):
        pass  # unreadable or malformed on disk -> rewrite it
    path.write_text(json.dumps(champions.to_dict(), indent=2, sort_keys=True) + "\n")
