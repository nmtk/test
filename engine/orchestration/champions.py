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

    One line per allocation, with its params compact. ``indent=2`` over a whole book is
    pathological here: a GP program is a deeply nested tree of ``["op", "mul", [...]]``
    nodes, and pretty-printing one puts every node on its own indented line. The first
    paper book measured **1,323 KB, of which 65 KB was content** — 95% indentation — and it
    is rewritten whenever a weight moves, which is every run. Written this way the same
    book is 65 KB and still greppable by allocation, which is how a human reads it; the
    readable view of what is allocated is the report's Champions table, not this file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists() and load_champions(path).allocations == champions.allocations:
            return
    except (OSError, ValueError, KeyError, TypeError):
        pass  # unreadable or malformed on disk -> rewrite it
    data = champions.to_dict()
    rows = [
        "    " + json.dumps(allocation, sort_keys=True, separators=(",", ":"))
        for allocation in data["allocations"]
    ]
    out = ["{", f'  "updated": {json.dumps(data["updated"])},', '  "allocations": [']
    out.append(",\n".join(rows))
    out += ["  ]", "}"]
    path.write_text("\n".join(line for line in out if line != "") + "\n")


# The paper executor trades a wider book than the live one: champions *plus* the
# gate-passers still incubating. See promotion.promote_step for why the rehearsal runs in
# parallel with incubation rather than after it. The live executor must never read this.
PAPER_CHAMPIONS_FILE = "paper_champions.json"


def paper_book_path(state_dir: Path) -> Path:
    """The paper executor's book, falling back to champions.json.

    The fallback matters for a state directory written before the rehearsal existed, and
    for `seed-champions`, which writes champions.json by hand.
    """
    paper = Path(state_dir) / PAPER_CHAMPIONS_FILE
    return paper if paper.is_file() else Path(state_dir) / "champions.json"
