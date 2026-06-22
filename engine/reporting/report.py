"""Generate a markdown status report from committed executor state.

Summarizes the current champions, positions, per-currency equity (with return and max
drawdown from the equity ledger), recent fills, and the strategy pipeline (incubating /
candidate counts). Written to ``state/reports/latest.md`` and a dated copy, so the
report itself is a versioned artifact.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine.core.config import Settings
from engine.orchestration import state as state_io
from engine.orchestration.champions import load_champions


def _max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def _equity_section(path: Path) -> list[str]:
    if not path.exists():
        return ["_no equity history yet_"]
    df = pd.read_parquet(path).sort_values("ts_ms")
    lines = [
        "| currency | start | current | return | max_dd | points |",
        "|---|---|---|---|---|---|",
    ]
    for currency, g in df.groupby("currency"):
        eq = g["equity"].reset_index(drop=True)
        start, current = float(eq.iloc[0]), float(eq.iloc[-1])
        ret = (current / start - 1.0) if start else 0.0
        lines.append(
            f"| {currency} | {start:,.0f} | {current:,.0f} | {ret:+.2%} | "
            f"{_max_drawdown(eq):.2%} | {len(eq)} |"
        )
    return lines


def _fills_section(path: Path, limit: int = 10) -> list[str]:
    if not path.exists():
        return ["_no fills yet_"]
    df = pd.read_parquet(path).tail(limit)
    lines = ["| time | symbol | side | qty | price | fee |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['fill_ts']} | {r['symbol']} | {r['side']} | {r['qty']:.6g} | "
            f"{r['price']:.6g} | {r['fee']:.4g} |"
        )
    return lines


def _pipeline_section(state_dir: Path) -> list[str]:
    """Show incubating strategies and candidate counts from population.db."""
    db_path = state_dir / "population.db"
    if not db_path.exists():
        return ["_no population yet — research has not run_"]

    import sqlite3

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM strategies ORDER BY status, family").fetchall()
        conn.close()
    except Exception:
        return ["_population.db unreadable_"]

    import json

    incubating = [r for r in rows if r["status"] == "incubating"]
    candidates = [r for r in rows if r["status"] == "candidate"]
    retired = [r for r in rows if r["status"] == "retired"]

    out: list[str] = [
        f"_incubating: {len(incubating)} | candidates: {len(candidates)} | "
        f"retired: {len(retired)}_",
        "",
    ]
    if not incubating:
        out.append("_no strategies in incubation (research gate has rejected all so far)_")
        return out

    out += [
        "| family | fwd_bars_needed | bt_sharpe | dsr | pbo |",
        "|---|---|---|---|---|",
    ]
    for r in incubating:
        metrics = json.loads(r["metrics"]) if isinstance(r["metrics"], str) else r["metrics"]
        start_ts = r["incubation_start_ts"] or 0
        start_dt = (
            datetime.fromtimestamp(start_ts / 1000, tz=UTC).strftime("%Y-%m-%d")
            if start_ts
            else "?"
        )
        bt_sharpe = metrics.get("sharpe_annual", float("nan"))
        dsr = metrics.get("dsr", float("nan"))
        pbo = metrics.get("pbo", float("nan"))
        out.append(
            f"| {r['family']} | 60 fwd bars (since {start_dt}) | "
            f"{bt_sharpe:.2f} | {dsr:.3f} | {pbo:.3f} |"
        )
    return out


def _trial_count(state_dir: Path) -> str:
    path = state_dir / "trial_ledger.json"
    if not path.exists():
        return "0"
    import json

    try:
        data = json.loads(path.read_text())
        total = sum(data.values()) if isinstance(data, dict) else 0
        return f"{total:,}"
    except Exception:
        return "?"


def generate_report(settings: Settings, mode: str = "paper") -> Path:
    champions = load_champions(settings.state_dir / "champions.json")
    account = state_io.load_account(state_io.account_path(settings.state_dir, mode))

    out: list[str] = [
        f"# trade — {mode} report",
        "",
        f"_generated {datetime.now(UTC).isoformat()} | "
        f"cumulative trials: {_trial_count(settings.state_dir)}_",
        "",
        f"## Champions ({len(champions.allocations)})",
    ]
    if champions.allocations:
        out += ["| strategy | symbol | tf | weight | params |", "|---|---|---|---|---|"]
        out += [
            f"| {a.strategy} | {a.symbol} | {a.timeframe} | {a.weight:+.3f} | {a.params} |"
            for a in champions.allocations
        ]
    else:
        out.append("_none — waiting for a strategy to clear research → incubation → promotion_")

    positions = account.get("positions", {})
    out += ["", "## Positions"]
    if positions:
        out += ["| symbol | qty | avg |", "|---|---|---|"]
        out += [f"| {s} | {d['qty']:.6g} | {d['avg']:.6g} |" for s, d in sorted(positions.items())]
    else:
        out.append("_flat_")

    out += ["", "## Equity", *_equity_section(state_io.equity_path(settings.state_dir, mode))]
    out += ["", "## Recent fills", *_fills_section(state_io.fills_path(settings.state_dir, mode))]
    out += ["", "## Pipeline", *_pipeline_section(settings.state_dir)]
    out.append("")

    text = "\n".join(out)
    reports_dir = settings.state_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "latest.md").write_text(text)
    dated = reports_dir / f"{datetime.now(UTC):%Y-%m-%d}-{mode}.md"
    dated.write_text(text)
    return reports_dir / "latest.md"
