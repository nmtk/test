"""Command-line entry point: ``trade <command>`` (also ``python -m engine``).

Commands: ``version``, ``status``, ``show-universe``, ``sync-data``, ``list-strategies``,
``backtest``, ``validate``, ``seed-champions``, ``paper-step``, ``live-step``, ``report``,
``research``, ``promote``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

import numpy as np

from engine import __version__
from engine.backtest import periods_per_year, run_backtest
from engine.core.config import load_settings, load_universe
from engine.core.killswitch import is_engaged
from engine.core.types import AssetClass
from engine.data.store import OHLCVStore
from engine.data.sync import sync_context, sync_universe
from engine.features.context import ContextAugmenter
from engine.orchestration.champions import Allocation, Champions, save_champions
from engine.orchestration.live import run_live_step
from engine.orchestration.orchestrator import run_step
from engine.orchestration.promotion import PromotionConfig, promote_step
from engine.reporting import generate_report
from engine.search.research import run_research
from engine.strategies import all_strategies, all_xs_strategies, get_strategy
from engine.strategies.base import FloatParam, IntParam
from engine.validation.gate import GateThresholds, evaluate_candidate_family
from engine.validation.ledger import TrialLedger


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"trade {__version__}")
    return 0


def _cmd_show_universe(_args: argparse.Namespace) -> int:
    settings = load_settings()
    universe = load_universe(settings.universe_path)
    print(f"{len(universe.entries)} instruments from {settings.universe_path}")
    for entry in universe.entries:
        inst = entry.instrument
        research = (
            f" research={entry.research_source.source}:{entry.research_source.symbol}"
            if entry.research_source
            else ""
        )
        print(
            f"  {inst.symbol:<10} {inst.asset_class.value:<7} venue={inst.venue.value:<8} "
            f"tf={','.join(entry.timeframes)}{research}"
        )
    return 0


def _cmd_macro_status(_args: argparse.Namespace) -> int:
    """Report which trade-flow series actually synced, and when each became public.

    The ``trade_balance`` archetype degrades to flat when its inputs are missing, which is
    safe but silent. This makes the silence visible: a retired FRED id shows up here as a
    missing series rather than as a strategy that mysteriously never trades.
    """
    import pandas as pd

    from engine.core.config import load_context_series
    from engine.features.macro_trade import TRADE_KEYS, availability_ts

    settings = load_settings()
    store = OHLCVStore(settings.data_dir / "ohlcv")
    series = [s for s in load_context_series(settings.context_path) if s.key in TRADE_KEYS]
    if not series:
        print("no trade-flow series configured in context.yaml")
        return 0

    print("=== trade-flow (macro) series ===")
    print(f"{'key':<12} {'symbol':<18} {'rows':>6} {'latest period':<15} {'public since':<15}")
    missing = 0
    for spec in series:
        df = store.read(spec.source, spec.symbol, spec.timeframe)
        if len(df) == 0:
            missing += 1
            print(f"{spec.key:<12} {spec.symbol:<18} {'-':>6} {'MISSING — check the FRED id':<31}")
            continue
        last_ts = df["ts"].iloc[-1:]
        period = pd.Timestamp(int(last_ts.iloc[0]), unit="ms", tz="UTC").date()
        public = pd.Timestamp(
            int(availability_ts(last_ts, spec.release_lag_days or 0).iloc[0]), unit="ms", tz="UTC"
        ).date()
        print(f"{spec.key:<12} {spec.symbol:<18} {len(df):>6} {str(period):<15} {str(public):<15}")

    if missing:
        print(
            f"\n{missing} series missing — trade_balance features read neutral and it stays flat."
        )
    return 0


def _cmd_sync_data(args: argparse.Namespace) -> int:
    settings = load_settings()
    universe = load_universe(settings.universe_path)
    store = OHLCVStore(settings.data_dir / "ohlcv")
    results = sync_universe(settings, universe, store, include_research=args.research)
    results.extend(sync_context(settings, store))

    failures = 0
    for r in results:
        if r.ok:
            print(f"  ok   {r.source:<8} {r.symbol:<10} {r.timeframe:<4} rows={r.rows_total}")
        else:
            failures += 1
            print(f"  FAIL {r.source:<8} {r.symbol:<10} {r.timeframe:<4} {r.error}")
    print(f"synced {len(results) - failures}/{len(results)} series")
    # Exit non-zero only if everything failed (partial success is fine for a cron job).
    return 1 if results and failures == len(results) else 0


def _cmd_list_strategies(_args: argparse.Namespace) -> int:
    print("single-asset:")
    for name, cls in sorted(all_strategies().items()):
        space = ", ".join(f"{k}={v}" for k, v in cls.param_space.items())
        print(f"  {name:<18} {space}")
    print("cross-sectional (panel):")
    for name, cls in sorted(all_xs_strategies().items()):
        space = ", ".join(f"{k}={v}" for k, v in cls.param_space.items())
        print(f"  {name:<18} {space}")
    return 0


def _parse_params(items: list[str] | None) -> dict:
    params: dict = {}
    for item in items or []:
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        try:
            params[key] = int(value)
        except ValueError:
            params[key] = float(value)
    return params


def _cmd_backtest(args: argparse.Namespace) -> int:
    settings = load_settings()
    universe = load_universe(settings.universe_path)
    entry = next((e for e in universe.entries if e.instrument.symbol == args.symbol), None)
    if entry is None and args.source is None:
        print(f"unknown symbol {args.symbol!r} in universe; pass --source explicitly")
        return 2
    source = args.source or entry.live_source.source
    asset_class = entry.instrument.asset_class if entry else AssetClass.CRYPTO

    store = OHLCVStore(settings.data_dir / "ohlcv")
    bars = store.read(source, args.symbol, args.timeframe)
    if len(bars) == 0:
        print(f"no data for {source}:{args.symbol}:{args.timeframe} — run sync-data first")
        return 1
    augmenter = ContextAugmenter.from_settings(settings, universe)
    bars = augmenter.augment(bars, source=source, symbol=args.symbol, timeframe=args.timeframe)

    strategy = get_strategy(args.strategy)(**_parse_params(args.param))
    weights = strategy.weights(bars)
    result = run_backtest(
        bars["close"],
        weights,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        periods_per_year=periods_per_year(args.timeframe, asset_class),
    )
    print(f"{strategy.describe()} on {source}:{args.symbol}:{args.timeframe} ({len(bars)} bars)")
    for key, value in result.stats.items():
        print(f"  {key:<14} {value:.4f}")
    return 0


def _grid_from_space(space: dict, n: int = 5) -> dict[str, list]:
    """Build a coarse parameter grid from a strategy's declared search space."""
    grid: dict[str, list] = {}
    for key, p in space.items():
        if isinstance(p, IntParam):
            grid[key] = sorted({int(round(x)) for x in np.linspace(p.low, p.high, n)})
        elif isinstance(p, FloatParam):
            grid[key] = [float(x) for x in np.linspace(p.low, p.high, n)]
        elif isinstance(p, list):
            grid[key] = list(p)
    return grid


def _cmd_validate(args: argparse.Namespace) -> int:
    settings = load_settings()
    universe = load_universe(settings.universe_path)
    entry = next((e for e in universe.entries if e.instrument.symbol == args.symbol), None)
    source = args.source or (entry.live_source.source if entry else None)
    if source is None:
        print(f"unknown symbol {args.symbol!r} in universe; pass --source explicitly")
        return 2
    asset_class = entry.instrument.asset_class if entry else AssetClass.CRYPTO

    store = OHLCVStore(settings.data_dir / "ohlcv")
    bars = store.read(source, args.symbol, args.timeframe)
    if len(bars) == 0:
        print(f"no data for {source}:{args.symbol}:{args.timeframe} — run sync-data first")
        return 1
    augmenter = ContextAugmenter.from_settings(settings, universe)
    bars = augmenter.augment(bars, source=source, symbol=args.symbol, timeframe=args.timeframe)

    strategy_cls = get_strategy(args.strategy)
    grid = _grid_from_space(strategy_cls.param_space, n=args.grid)
    ledger = TrialLedger.load(settings.state_dir / "trial_ledger.json")
    report = evaluate_candidate_family(
        bars,
        strategy_cls,
        grid,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        ppy=periods_per_year(args.timeframe, asset_class),
        thresholds=GateThresholds(),
        ledger=ledger,
        family=f"{args.strategy}:{args.symbol}:{args.timeframe}",
    )
    ledger.save()

    print(f"validation: {report.name}  best={report.best_params}")
    print(f"  bars={report.n_bars} configs={report.n_configs} cumulative_trials={report.n_trials}")
    for key in ("sharpe_annual", "cpcv_median", "cpcv_pct_positive", "psr", "dsr", "pbo", "spa_p"):
        print(f"  {key:<18} {getattr(report, key):.4f}")
    print(f"  mintrl(bars)       {report.mintrl:.0f}")
    for check, ok in report.checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {check}")
    print(f"  => {'ACCEPTED as challenger' if report.passed else 'REJECTED'}")
    return 0


def _cmd_seed_champions(args: argparse.Namespace) -> int:
    settings = load_settings()
    universe = load_universe(settings.universe_path)
    if args.symbol:
        entries = [e for e in universe.entries if e.instrument.symbol in set(args.symbol)]
    else:
        entries = [e for e in universe.entries if e.instrument.asset_class is AssetClass.CRYPTO]
    if not entries:
        print("no matching instruments in universe")
        return 1

    params = _parse_params(args.param)
    weight = args.weight if args.weight is not None else round(1.0 / len(entries), 4)
    allocations = [
        Allocation(
            strategy=args.strategy,
            symbol=e.instrument.symbol,
            source=e.live_source.source,
            timeframe=args.timeframe,
            weight=weight,
            params=params,
        )
        for e in entries
    ]
    champions = Champions(allocations=allocations, updated=datetime.now(UTC).isoformat())
    path = settings.state_dir / "champions.json"
    save_champions(path, champions)
    print(f"seeded {len(allocations)} champions ({args.strategy} @ weight {weight}) -> {path}")
    return 0


def _cmd_paper_step(args: argparse.Namespace) -> int:
    settings = load_settings()
    result = run_step(settings, mode=args.mode)
    print(
        f"[{result.mode}] bar_ts={result.bar_ts_ms} orders={result.n_orders} "
        f"killed={result.killed} note={result.note}"
    )
    for cur, val in sorted(result.equity.items()):
        print(f"  equity {cur}: {val:,.2f}")
    for sym, qty in sorted(result.positions.items()):
        print(f"  pos {sym}: {qty:.6g}")
    return 0


def _cmd_live_step(_args: argparse.Namespace) -> int:
    settings = load_settings()
    result = run_live_step(settings)
    print(
        f"[live] enabled={settings.live_enabled} bar_ts={result.bar_ts_ms} "
        f"orders={result.n_orders} killed={result.killed} note={result.note}"
    )
    for cur, val in sorted(result.equity.items()):
        print(f"  equity {cur}: {val:,.2f}")
    for sym, qty in sorted(result.positions.items()):
        print(f"  pos {sym}: {qty:.6g}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    settings = load_settings()
    path = generate_report(settings, mode=args.mode)
    print(f"wrote {path}")
    return 0


def _cmd_research(args: argparse.Namespace) -> int:
    settings = load_settings()
    summary = run_research(
        settings,
        strategies=tuple(args.strategy) if args.strategy else None,
        symbols=args.symbol,
        timeframes=args.timeframe,
        trials=args.trials,
        include_gp=args.gp,
    )
    print(
        f"research: evaluated={summary.n_evaluated} passed={summary.n_passed} "
        f"(passed -> incubating; run `trade promote`)"
    )
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    settings = load_settings()
    summary = promote_step(settings, cfg=PromotionConfig(max_champions=args.max_champions))
    print(
        f"promote: incubating={summary.n_incubating} promoted={summary.n_promoted} "
        f"champions_checked={summary.n_champions_checked} retired={summary.n_retired} "
        f"champions={summary.n_champions}"
    )
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    import json
    import sqlite3

    settings = load_settings()
    universe = load_universe(settings.universe_path)

    print("=== trade system status ===")
    ks = is_engaged(settings.state_dir)
    print(f"kill_switch : {'ON  ← ALL EXECUTION HALTED' if ks else 'off'}")
    print(f"live_enabled: {settings.live_enabled}")

    # Trial count
    ledger_path = settings.state_dir / "trial_ledger.json"
    if ledger_path.exists():
        try:
            data = json.loads(ledger_path.read_text())
            total = sum(data.values()) if isinstance(data, dict) else 0
            print(f"trials      : {total:,} cumulative (raises DSR bar on each research run)")
        except Exception:
            pass

    # Data freshness
    store = OHLCVStore(settings.data_dir / "ohlcv")
    print("\ndata freshness:")
    for entry in universe.entries:
        sym = entry.instrument.symbol
        src = entry.live_source.source
        for tf in entry.timeframes:
            ts = store.last_ts(src, sym, tf)
            if ts:
                dt = datetime.fromtimestamp(ts / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%MZ")
                bars = store.read(src, sym, tf)
                print(f"  {src:<8} {sym:<10} {tf:<4}  last={dt}  bars={len(bars)}")
            else:
                print(f"  {src:<8} {sym:<10} {tf:<4}  (no data — run sync-data)")

    # Population pipeline
    db_path = settings.state_dir / "population.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            counts = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT status, COUNT(*) FROM strategies GROUP BY status"
                ).fetchall()
            }
            conn.close()
            print("\npopulation:")
            for status in ("champion", "incubating", "candidate", "retired"):
                n = counts.get(status, 0)
                print(f"  {status:<12} {n}")
        except Exception:
            pass
    else:
        print("\npopulation: (no population.db — research has not run yet)")

    # Current champions
    from engine.orchestration.champions import load_champions

    champs = load_champions(settings.state_dir / "champions.json")
    print(f"\nchampions ({len(champs.allocations)}):")
    if champs.allocations:
        for a in champs.allocations:
            print(f"  {a.strategy:<18} {a.symbol:<12} tf={a.timeframe}  weight={a.weight:+.4f}")
    else:
        print("  (none — waiting for incubation to complete)")

    # Latest report
    latest = settings.state_dir / "reports" / "latest.md"
    if latest.exists():
        mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=UTC)
        print(f"\nlatest report: {latest}  ({mtime:%Y-%m-%dT%H:%MZ})")
    else:
        print("\nlatest report: not yet written (executor-paper must run first)")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trade", description="multi-asset trading research engine"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)
    sub.add_parser(
        "status", help="show system health: data freshness, pipeline, champions, kill switch"
    ).set_defaults(func=_cmd_status)
    sub.add_parser(
        "macro-status", help="show which trade-flow series synced and when each went public"
    ).set_defaults(func=_cmd_macro_status)
    sub.add_parser("show-universe", help="print the configured universe").set_defaults(
        func=_cmd_show_universe
    )

    p_sync = sub.add_parser("sync-data", help="incrementally fetch OHLCV into the store")
    p_sync.add_argument(
        "--research", action="store_true", help="also sync deep-history research sources"
    )
    p_sync.set_defaults(func=_cmd_sync_data)

    sub.add_parser("list-strategies", help="list registered seed strategies").set_defaults(
        func=_cmd_list_strategies
    )

    p_bt = sub.add_parser("backtest", help="backtest a seed strategy over stored data")
    p_bt.add_argument("--strategy", required=True, help="strategy name (see list-strategies)")
    p_bt.add_argument("--symbol", required=True, help="canonical symbol, e.g. BTC/JPY")
    p_bt.add_argument("--timeframe", default="1d")
    p_bt.add_argument("--source", default=None, help="data source (default: from universe)")
    p_bt.add_argument("--param", action="append", help="strategy param, e.g. lookback=50")
    p_bt.add_argument("--fee-bps", type=float, default=10.0)
    p_bt.add_argument("--slippage-bps", type=float, default=5.0)
    p_bt.set_defaults(func=_cmd_backtest)

    p_seed = sub.add_parser("seed-champions", help="bootstrap a champions.json to paper-trade")
    p_seed.add_argument("--strategy", required=True, help="strategy name (see list-strategies)")
    p_seed.add_argument(
        "--symbol", action="append", help="symbol to include (repeatable; default: all crypto)"
    )
    p_seed.add_argument("--timeframe", default="1h")
    p_seed.add_argument(
        "--weight", type=float, default=None, help="per-symbol weight (default 1/N)"
    )
    p_seed.add_argument("--param", action="append", help="strategy param, e.g. lookback=100")
    p_seed.set_defaults(func=_cmd_seed_champions)

    p_paper = sub.add_parser("paper-step", help="run one paper executor step")
    p_paper.add_argument("--mode", default="paper")
    p_paper.set_defaults(func=_cmd_paper_step)

    sub.add_parser(
        "live-step", help="run one LIVE executor step (no-op unless live is enabled + keys set)"
    ).set_defaults(func=_cmd_live_step)

    p_report = sub.add_parser("report", help="generate the markdown status report")
    p_report.add_argument("--mode", default="paper")
    p_report.set_defaults(func=_cmd_report)

    p_research = sub.add_parser("research", help="search -> validate -> incubate")
    p_research.add_argument("--strategy", action="append", help="archetype (repeatable)")
    p_research.add_argument("--symbol", action="append", help="symbol (repeatable)")
    p_research.add_argument("--timeframe", action="append", help="timeframe (repeatable)")
    p_research.add_argument("--trials", type=int, default=40, help="optuna trials per family")
    p_research.add_argument("--gp", action="store_true", help="also run GP alpha discovery")
    p_research.set_defaults(func=_cmd_research)

    p_promote = sub.add_parser(
        "promote", help="incubation/decay FSM: promote validated strategies, retire decayed"
    )
    p_promote.add_argument("--max-champions", type=int, default=8)
    p_promote.set_defaults(func=_cmd_promote)

    p_val = sub.add_parser("validate", help="run the validation gate on a strategy family")
    p_val.add_argument("--strategy", required=True, help="strategy name (see list-strategies)")
    p_val.add_argument("--symbol", required=True, help="canonical symbol, e.g. BTC/JPY")
    p_val.add_argument("--timeframe", default="1d")
    p_val.add_argument("--source", default=None, help="data source (default: from universe)")
    p_val.add_argument("--grid", type=int, default=5, help="grid points per parameter")
    p_val.add_argument("--fee-bps", type=float, default=10.0)
    p_val.add_argument("--slippage-bps", type=float, default=5.0)
    p_val.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
