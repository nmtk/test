"""Command-line entry point: ``trade <command>`` (also ``python -m engine``).

Implemented now (M0): ``version``, ``show-universe``, ``sync-data``.
Stubs for later milestones: ``paper-step`` (M3), ``research``/``validate`` (M2/M4/M5),
``report`` (M3). Stubs exit 0 with a clear message so Makefile/E2E wiring works early.
"""

from __future__ import annotations

import argparse
import sys

from engine import __version__
from engine.backtest import periods_per_year, run_backtest
from engine.core.config import load_settings, load_universe
from engine.core.types import AssetClass
from engine.data.store import OHLCVStore
from engine.data.sync import sync_universe
from engine.strategies import all_strategies, all_xs_strategies, get_strategy


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


def _cmd_sync_data(args: argparse.Namespace) -> int:
    settings = load_settings()
    universe = load_universe(settings.universe_path)
    store = OHLCVStore(settings.data_dir / "ohlcv")
    results = sync_universe(settings, universe, store, include_research=args.research)

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


def _cmd_stub(name: str, milestone: str):
    def run(_args: argparse.Namespace) -> int:
        print(f"[{name}] not implemented yet (planned for {milestone})")
        return 0

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trade", description="multi-asset trading research engine"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)
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

    sub.add_parser("paper-step", help="(M3) one paper executor step").set_defaults(
        func=_cmd_stub("paper-step", "M3")
    )
    sub.add_parser("research", help="(M4/M5) nightly search + GP discovery").set_defaults(
        func=_cmd_stub("research", "M4/M5")
    )
    sub.add_parser("validate", help="(M2) run the validation gate").set_defaults(
        func=_cmd_stub("validate", "M2")
    )
    sub.add_parser("report", help="(M3) generate the markdown report").set_defaults(
        func=_cmd_stub("report", "M3")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
