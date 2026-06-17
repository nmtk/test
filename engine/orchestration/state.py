"""Persistence for the executor: account snapshot (JSON), fills + equity ledgers (Parquet).

Keeping the account as committed state is what lets a stateless cron run resume: each run
restores cash/positions, acts, and writes them back. Fills and equity are append-only audit
logs (the git history of these files is the trading record).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from engine.execution.paper import PaperBroker


def account_path(state_dir: Path, mode: str) -> Path:
    return Path(state_dir) / mode / "account.json"


def fills_path(state_dir: Path, mode: str) -> Path:
    return Path(state_dir) / mode / "fills.parquet"


def equity_path(state_dir: Path, mode: str) -> Path:
    return Path(state_dir) / mode / "equity.parquet"


def load_account(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_account(path: Path, broker: PaperBroker) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(broker.snapshot(), indent=2, sort_keys=True) + "\n")


def append_fills(path: Path, fills: list, ts_ms: int) -> None:
    """Append this step's fills to the parquet ledger."""
    if not fills:
        return
    path = Path(path)
    rows = [
        {
            "ts_ms": ts_ms,
            "fill_ts": f.ts.isoformat(),
            "symbol": f.symbol,
            "side": str(f.side),
            "qty": f.qty,
            "price": f.price,
            "fee": f.fee,
            "fee_currency": f.fee_currency,
            "venue": str(f.venue),
            "client_order_id": f.client_order_id,
        }
        for f in fills
    ]
    _append_parquet(path, pd.DataFrame(rows))


def append_equity(path: Path, ts_ms: int, equity: dict[str, float]) -> None:
    """Append a mark-to-market equity snapshot (one row per currency)."""
    path = Path(path)
    now = datetime.now(UTC).isoformat()
    rows = [
        {"ts_ms": ts_ms, "at": now, "currency": cur, "equity": val} for cur, val in equity.items()
    ]
    _append_parquet(path, pd.DataFrame(rows))


def _append_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, engine="pyarrow", index=False)
