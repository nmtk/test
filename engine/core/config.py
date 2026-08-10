"""Runtime configuration: filesystem paths, secrets (from env), and the universe.

Secrets are **only** read from environment variables — never committed. Locally you may
keep them in a ``.env`` file (git-ignored); in CI they come from Actions Secrets. API
keys must be created **without withdrawal permission**.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from engine.core.types import AssetClass, Instrument, Venue


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (or this file) to the dir containing pyproject.toml."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: two levels up from engine/core/config.py == repo root.
    return Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Does not overwrite existing env vars."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True, slots=True)
class BrokerCredentials:
    api_key: str | None
    api_secret: str | None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)


@dataclass(frozen=True, slots=True)
class Settings:
    repo_root: Path
    data_dir: Path
    state_dir: Path
    config_dir: Path
    live_enabled: bool
    alpaca_paper: bool
    bitbank: BrokerCredentials
    alpaca: BrokerCredentials

    @property
    def universe_path(self) -> Path:
        return self.config_dir / "universe.yaml"

    @property
    def context_path(self) -> Path:
        return self.config_dir / "context.yaml"


def load_settings(repo_root: Path | None = None) -> Settings:
    root = (repo_root or _find_repo_root()).resolve()
    _load_dotenv(root / ".env")
    return Settings(
        repo_root=root,
        data_dir=root / "data",
        state_dir=root / "state",
        config_dir=root / "strategies_config",
        # Live trading is OFF unless explicitly enabled. This is a safety default.
        live_enabled=_env_bool("TRADE_LIVE_ENABLED", default=False),
        alpaca_paper=_env_bool("ALPACA_PAPER", default=True),
        bitbank=BrokerCredentials(
            api_key=os.environ.get("BITBANK_API_KEY"),
            api_secret=os.environ.get("BITBANK_API_SECRET"),
        ),
        alpaca=BrokerCredentials(
            api_key=os.environ.get("ALPACA_API_KEY"),
            api_secret=os.environ.get("ALPACA_API_SECRET"),
        ),
    )


# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DataSourceSpec:
    """Where to fetch bars for an instrument. ``source`` is a data-layer source name."""

    source: str  # "bitbank" | "binance" | "alpaca"
    symbol: str


@dataclass(frozen=True, slots=True)
class UniverseEntry:
    instrument: Instrument
    timeframes: tuple[str, ...]
    live_source: DataSourceSpec
    research_source: DataSourceSpec | None = None


@dataclass(frozen=True, slots=True)
class Universe:
    entries: tuple[UniverseEntry, ...] = field(default_factory=tuple)

    def by_asset_class(self, asset_class: AssetClass) -> list[UniverseEntry]:
        return [e for e in self.entries if e.instrument.asset_class is asset_class]


_DEFAULT_SOURCE_BY_VENUE = {
    Venue.BITBANK: "bitbank",
    Venue.ALPACA: "alpaca",
}


def load_universe(path: Path, default_timeframes: tuple[str, ...] = ("1h", "4h", "1d")) -> Universe:
    """Load the instrument universe from a YAML file.

    See ``strategies_config/universe.yaml`` for the schema.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    file_timeframes = tuple(raw.get("timeframes") or default_timeframes)

    entries: list[UniverseEntry] = []
    for item in raw.get("instruments", []):
        asset_class = AssetClass(item["asset_class"])
        venue = Venue(item["venue"])
        instrument = Instrument(
            symbol=item["symbol"],
            asset_class=asset_class,
            venue=venue,
            base=item.get("base"),
            quote=item.get("quote"),
            min_size=float(item.get("min_size", 0.0)),
            size_step=float(item.get("size_step", 0.0)),
            price_step=float(item.get("price_step", 0.0)),
            min_notional=float(item.get("min_notional", 0.0)),
        )
        timeframes = tuple(item.get("timeframes") or file_timeframes)
        live_source = DataSourceSpec(
            source=_DEFAULT_SOURCE_BY_VENUE[venue],
            symbol=item["symbol"],
        )
        research_source = None
        if "research" in item and item["research"]:
            research_source = DataSourceSpec(
                source=item["research"]["source"],
                symbol=item["research"]["symbol"],
            )
        entries.append(
            UniverseEntry(
                instrument=instrument,
                timeframes=timeframes,
                live_source=live_source,
                research_source=research_source,
            )
        )
    return Universe(entries=tuple(entries))


# --------------------------------------------------------------------------------------
# Cross-asset context series (macro flow layer)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContextSeries:
    """One external series the flow layer conditions on (see engine/features/context.py).

    ``key`` is the feature namespace (e.g. ``spx`` -> ``ctx_spx_mom``); ``source`` /
    ``symbol`` say where to fetch it. Context series are research *inputs*, never traded.
    """

    key: str
    source: str  # "stooq" | "fred"
    symbol: str
    timeframe: str = "1d"
    # Days between the END of the observation period and publication. Only meaningful for
    # released statistics (trade data); market prices are known instantly and leave it
    # None. See engine/features/macro_trade.py for why this must not be omitted there.
    release_lag_days: int | None = None


def load_context_series(path: Path) -> list[ContextSeries]:
    """Load the context universe from YAML; a missing file just means no context."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[ContextSeries] = []
    for item in raw.get("series", []):
        out.append(
            ContextSeries(
                key=str(item["key"]),
                source=str(item["source"]),
                symbol=str(item["symbol"]),
                timeframe=str(item.get("timeframe", "1d")),
                release_lag_days=(
                    int(item["release_lag_days"]) if "release_lag_days" in item else None
                ),
            )
        )
    return out
