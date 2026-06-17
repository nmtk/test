"""The strategy population: a committed SQLite registry of every candidate ever evaluated.

Rows move through statuses (candidate -> challenger -> champion -> retired) as the nightly
search and (later) incubation promote or retire them. Keeping this as committed state makes
the evolution of the book an auditable git history. Stdlib ``sqlite3`` only.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family TEXT NOT NULL,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    params TEXT NOT NULL,
    status TEXT NOT NULL,
    metrics TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(family, params)
);
"""


@dataclass(slots=True)
class StrategyRecord:
    family: str
    strategy: str
    symbol: str
    source: str
    timeframe: str
    params: dict
    status: str
    metrics: dict


class Population:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def upsert(self, record: StrategyRecord) -> None:
        now = datetime.now(UTC).isoformat()
        params = json.dumps(record.params, sort_keys=True)
        metrics = json.dumps(record.metrics, sort_keys=True)
        self.conn.execute(
            """
            INSERT INTO strategies
                (family, strategy, symbol, source, timeframe, params, status, metrics,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(family, params) DO UPDATE SET
                status=excluded.status, metrics=excluded.metrics, updated_at=excluded.updated_at
            """,
            (
                record.family,
                record.strategy,
                record.symbol,
                record.source,
                record.timeframe,
                params,
                record.status,
                metrics,
                now,
                now,
            ),
        )
        self.conn.commit()

    def list(self, status: str | None = None) -> list[dict]:
        if status is None:
            rows = self.conn.execute("SELECT * FROM strategies ORDER BY id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM strategies WHERE status = ? ORDER BY id", (status,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def set_status(self, family: str, params: dict, status: str) -> None:
        self.conn.execute(
            "UPDATE strategies SET status = ?, updated_at = ? WHERE family = ? AND params = ?",
            (status, datetime.now(UTC).isoformat(), family, json.dumps(params, sort_keys=True)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["params"] = json.loads(d["params"])
        d["metrics"] = json.loads(d["metrics"])
        return d
