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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(champions.to_dict(), indent=2, sort_keys=True) + "\n")
