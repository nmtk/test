"""Generate a markdown status report from committed executor state.

Summarizes the current champions, positions, per-currency equity (with return and max
drawdown from the equity ledger), and recent fills. Written to ``state/reports/latest.md``
and a dated copy, so the report itself is a versioned artifact.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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


def generate_report(settings: Settings, mode: str = "paper") -> Path:
    champions = load_champions(settings.state_dir / "champions.json")
    account = state_io.load_account(state_io.account_path(settings.state_dir, mode))

    out: list[str] = [
        f"# trade — {mode} report",
        "",
        f"_generated {datetime.now(UTC).isoformat()}_",
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
        out.append("_none — run `trade seed-champions` or wait for research (M4)_")

    positions = account.get("positions", {})
    out += ["", "## Positions"]
    if positions:
        out += ["| symbol | qty | avg |", "|---|---|---|"]
        out += [f"| {s} | {d['qty']:.6g} | {d['avg']:.6g} |" for s, d in sorted(positions.items())]
    else:
        out.append("_flat_")

    out += ["", "## Equity", *_equity_section(state_io.equity_path(settings.state_dir, mode))]
    out += ["", "## Recent fills", *_fills_section(state_io.fills_path(settings.state_dir, mode))]
    out.append("")

    text = "\n".join(out)
    reports_dir = settings.state_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "latest.md").write_text(text)
    dated = reports_dir / f"{datetime.now(UTC):%Y-%m-%d}-{mode}.md"
    dated.write_text(text)
    return reports_dir / "latest.md"
