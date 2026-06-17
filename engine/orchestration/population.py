"""The strategy population: a committed SQLite registry, one row per family.

A *family* is ``archetype:symbol:timeframe`` — you run at most one parameterization of it at
a time, so it is the natural key. Rows move through statuses as the search discovers and the
promotion FSM incubates/promotes/retires them:

    candidate   -> failed the backtest validation gate
    incubating  -> passed the gate; being shadow-tested on genuinely forward data
    champion    -> passed incubation; allocated and paper-traded
    retired     -> decayed out of being a champion

Keeping this as committed state makes the evolution of the book an auditable git history.
Stdlib ``sqlite3`` only.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_COLUMNS = (
    "strategy",
    "symbol",
    "source",
    "timeframe",
    "params",
    "status",
    "metrics",
    "incubation_start_ts",
    "champion_start_ts",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    family TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    params TEXT NOT NULL,
    status TEXT NOT NULL,
    metrics TEXT NOT NULL,
    incubation_start_ts INTEGER,
    champion_start_ts INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
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
    incubation_start_ts: int | None = None
    champion_start_ts: int | None = None


class Population:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def get(self, family: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM strategies WHERE family = ?", (family,)).fetchone()
        return self._row_to_dict(row) if row else None

    def upsert(self, record: StrategyRecord) -> None:
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """
            INSERT INTO strategies
                (family, strategy, symbol, source, timeframe, params, status, metrics,
                 incubation_start_ts, champion_start_ts, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(family) DO UPDATE SET
                strategy=excluded.strategy, symbol=excluded.symbol, source=excluded.source,
                timeframe=excluded.timeframe, params=excluded.params, status=excluded.status,
                metrics=excluded.metrics, incubation_start_ts=excluded.incubation_start_ts,
                champion_start_ts=excluded.champion_start_ts, updated_at=excluded.updated_at
            """,
            (
                record.family,
                record.strategy,
                record.symbol,
                record.source,
                record.timeframe,
                json.dumps(record.params, sort_keys=True),
                record.status,
                json.dumps(record.metrics, sort_keys=True),
                record.incubation_start_ts,
                record.champion_start_ts,
                now,
                now,
            ),
        )
        self.conn.commit()

    def list(self, status: str | None = None) -> list[dict]:
        if status is None:
            rows = self.conn.execute("SELECT * FROM strategies ORDER BY family").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM strategies WHERE status = ? ORDER BY family", (status,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_fields(self, family: str, **fields) -> None:
        """Update whitelisted columns for a family (e.g. status, champion_start_ts)."""
        cols = [c for c in fields if c in _COLUMNS]
        if not cols:
            return
        assignments = ", ".join(f"{c} = ?" for c in cols)
        values = [
            json.dumps(fields[c], sort_keys=True) if c in ("params", "metrics") else fields[c]
            for c in cols
        ]
        self.conn.execute(
            f"UPDATE strategies SET {assignments}, updated_at = ? WHERE family = ?",
            (*values, datetime.now(UTC).isoformat(), family),
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
