"""The trial ledger: an honest, cumulative count of every candidate ever tried.

The Deflated Sharpe Ratio must be deflated by the *total* number of trials attempted over
the program's life — not just tonight's — or it systematically understates overfitting.
This persists those counts (per family) as committed JSON state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrialLedger:
    path: Path
    counts: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> TrialLedger:
        path = Path(path)
        counts: dict[str, int] = {}
        if path.exists():
            counts = {str(k): int(v) for k, v in json.loads(path.read_text()).items()}
        return cls(path=path, counts=counts)

    def get(self, family: str) -> int:
        return self.counts.get(family, 0)

    def total(self) -> int:
        return sum(self.counts.values())

    def add(self, family: str, n: int) -> int:
        """Add ``n`` trials to ``family`` and return the new cumulative count."""
        self.counts[family] = self.get(family) + int(n)
        return self.counts[family]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.counts, indent=2, sort_keys=True) + "\n")
