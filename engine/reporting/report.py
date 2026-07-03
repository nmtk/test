"""Generate a markdown status report from committed executor state.

Summarizes the current champions, positions, per-currency equity (with return and max
drawdown from the equity ledger), recent fills, the strategy pipeline, and a per-archetype
performance breakdown with plain-language strategy descriptions.
Written to ``state/reports/latest.md`` and a dated copy.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from engine.core.config import Settings
from engine.orchestration import state as state_io
from engine.orchestration.champions import load_champions

# Plain-language descriptions of each strategy archetype (Japanese).
_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "tsmom": (
        "時系列モメンタム",
        "過去 N 本のリターン符号でポジションを決める。"
        "上昇トレンドが続いていれば買い、下降トレンドなら売り。"
        "「トレンドは持続する」という仮説に基づく最もシンプルで実績豊富なファクター。"
        "パラメータ: `lookback`（参照期間）",
    ),
    "ma_cross": (
        "移動平均クロス",
        "短期移動平均（fast）が長期移動平均（slow）を上抜けたら買い、下抜けたら売り。"
        "MA クロスはトレンド転換の代表的なシグナル。"
        "パラメータ: `fast`（短期 MA）, `slow`（長期 MA）",
    ),
    "donchian": (
        "ドンチアンブレイクアウト",
        "過去 N 本の最高値を更新したら買い、最安値を更新したら売る（前バー確定値で判断）。"
        "「新高値は続騰のサイン」というチャネルブレイクアウト戦略の原型。"
        "パラメータ: `lookback`（チャネル期間）",
    ),
    "zscore_reversion": (
        "Z スコア平均回帰",
        "価格が移動平均から Z 標準偏差以上離れたら逆張り（売られすぎ→買い、買われすぎ→売り）。"
        "ボリンジャーバンドと同じ発想。モメンタムと逆方向に張るため、レジーム依存が強い。"
        "パラメータ: `window`（移動平均期間）, `z`（発動閾値）",
    ),
    "macro_flow": (
        "マクロフロー・ゲート付きトレンド",
        "対象銘柄のトレンドに乗るのは「資金が流れ込みやすい地合い」のときだけ、という戦略。"
        "株式（S&P500）・金・米金利・ドル・BTC 市場全体・対 BTC 相対力などのクロスアセット"
        "指標（ctx_* 特徴量、全て前日以前の確定値のみ使用）をゲートとして使い、"
        "ゲート条件を満たす間だけトレンドシグナルでポジションを持つ。"
        "パラメータ: `lookback`（トレンド期間）, `gate`（どの資金フロー指標か）, "
        "`direction`（順方向/逆方向のどちらの仮説か）, `threshold`（発動閾値）",
    ),
    "xs_momentum": (
        "横断モメンタム（クロスセクション）",
        "複数銘柄を横並びで比較し、相対的に強い銘柄を買い・弱い銘柄を売るドル中立戦略。"
        "単一銘柄のトレンドではなく「他と比べてどうか」で判断する。"
        "パラメータ: `lookback`（ランキング期間）",
    ),
    "gp": (
        "GP 自動発見式（遺伝的プログラミング）",
        "遺伝的プログラミングが特徴量の組み合わせを自動進化させて発見した実験的な数式。"
        "特徴量は価格・出来高（リターン・HL 幅・出来高 Z など）に加え、クロスアセットの"
        "資金フロー指標（S&P500・金・米金利・ドル・BTC 市場全体との相対関係などの ctx_* 系列）"
        "も端子として使える。人間が設計したルールではない。"
        "過剰適合リスクが最も高いため、検証ゲートが特に厳格（DSR ≥ 0.99、PBO ≤ 0.10）。",
    ),
}

_STATUS_LABEL = {
    "champion": "champion",
    "incubating": "incubating",
    "candidate": "rejected",
    "retired": "retired",
}

# Gate thresholds (for the footnote reminder).
_GATE = "合格基準: DSR ≥ 0.95 / PBO ≤ 0.20 / SPA p ≤ 0.05 / CPCV 中央値 ≥ 0 / CPCV 正率 ≥ 50%"


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
    db_path = state_dir / "population.db"
    if not db_path.exists():
        return ["_no population yet — research has not run_"]
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM strategies ORDER BY status, family").fetchall()
        conn.close()
    except Exception:
        return ["_population.db unreadable_"]

    incubating = [r for r in rows if r["status"] == "incubating"]
    candidates = [r for r in rows if r["status"] == "candidate"]
    retired = [r for r in rows if r["status"] == "retired"]

    out: list[str] = [
        f"_incubating: {len(incubating)} | rejected: {len(candidates)} | retired: {len(retired)}_",
        "",
    ]
    if not incubating:
        out.append("_養生中の戦略なし（研究ゲートがまだ合格を出していない）_")
        return out

    out += [
        "| family | since | bt_sharpe | dsr | pbo |",
        "|---|---|---|---|---|",
    ]
    for r in incubating:
        m = json.loads(r["metrics"]) if isinstance(r["metrics"], str) else r["metrics"]
        start_ts = r["incubation_start_ts"] or 0
        since = (
            datetime.fromtimestamp(start_ts / 1000, tz=UTC).strftime("%Y-%m-%d")
            if start_ts
            else "?"
        )
        out.append(
            f"| {r['family']} | {since} | "
            f"{m.get('sharpe_annual', float('nan')):.2f} | "
            f"{m.get('dsr', float('nan')):.3f} | "
            f"{m.get('pbo', float('nan')):.3f} |"
        )
    return out


def _performance_section(state_dir: Path) -> list[str]:
    """Per-archetype results table + plain-language strategy description."""
    db_path = state_dir / "population.db"
    if not db_path.exists():
        return ["_research has not run yet_"]
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM strategies ORDER BY strategy, symbol, timeframe"
        ).fetchall()
        conn.close()
    except Exception:
        return ["_population.db unreadable_"]

    if not rows:
        return ["_no strategies evaluated yet_"]

    by_archetype: dict[str, list] = defaultdict(list)
    for r in rows:
        by_archetype[r["strategy"]].append(r)

    out: list[str] = []
    # Show known archetypes first in a fixed order, then any unknowns.
    order = [
        "tsmom",
        "ma_cross",
        "donchian",
        "zscore_reversion",
        "macro_flow",
        "xs_momentum",
        "gp",
    ]
    archetypes = order + [a for a in sorted(by_archetype) if a not in order]

    for archetype in archetypes:
        if archetype not in by_archetype:
            continue
        records = by_archetype[archetype]

        name_jp, desc = _DESCRIPTIONS.get(archetype, (archetype, ""))
        out.append(f"### {archetype} — {name_jp}")
        out.append("")
        if desc:
            out.append(desc)
            out.append("")

        # Sort by DSR descending so the best attempt appears first.
        def _dsr(r) -> float:
            m = json.loads(r["metrics"]) if isinstance(r["metrics"], str) else r["metrics"]
            return m.get("dsr", 0.0)

        records = sorted(records, key=_dsr, reverse=True)

        out += [
            "| symbol | tf | bt_sharpe | max_dd | dsr | pbo | spa_p | cpcv_med | 結果 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in records:
            m = json.loads(r["metrics"]) if isinstance(r["metrics"], str) else r["metrics"]
            label = _STATUS_LABEL.get(r["status"], r["status"])
            out.append(
                f"| {r['symbol']} | {r['timeframe']} | "
                f"{m.get('sharpe_annual', float('nan')):+.2f} | "
                f"{m.get('max_drawdown', float('nan')):+.2%} | "
                f"{m.get('dsr', float('nan')):.3f} | "
                f"{m.get('pbo', float('nan')):.3f} | "
                f"{m.get('spa_p', float('nan')):.3f} | "
                f"{m.get('cpcv_median', float('nan')):+.3f} | "
                f"**{label}** |"
            )

        out += ["", f"_{_GATE}_", ""]

    return out


def _trial_count(state_dir: Path) -> str:
    path = state_dir / "trial_ledger.json"
    if not path.exists():
        return "0"
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
    out += ["", "## Strategy Performance", *_performance_section(settings.state_dir)]
    out.append("")

    text = "\n".join(out)
    reports_dir = settings.state_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "latest.md").write_text(text)
    dated = reports_dir / f"{datetime.now(UTC):%Y-%m-%d}-{mode}.md"
    dated.write_text(text)
    return reports_dir / "latest.md"
