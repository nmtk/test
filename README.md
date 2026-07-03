# trade — self-improving multi-asset trading research engine

An automated trading system that does **not** rely on a single fixed strategy.
Its core is a *research engine with a rigorous validation gate*: it continuously
searches for, validates, and rotates strategies across **US equities (Alpaca)**
and **crypto (bitbank)** — one engine, multiple brokers.

> **Honest premise:** there is no single "most profitable" strategy, and any known
> edge decays as it gets crowded. The only durable approach is a *factory that keeps
> looking* — and that guards itself against overfitting, which is the real enemy.

## Design in one picture

- **Stable code = the engine.** Search, validation, execution, risk, monitoring.
  Changes rarely.
- **Evolving data = the strategy population.** Champions, challengers, and
  GP-discovered formulas are *data* the bot commits to `state/`. They evolve nightly;
  the code does not.
- **Cross-asset context = the flow layer.** Free keyless macro series (S&P 500, gold,
  US 10y, USD/JPY, EUR/USD via stooq/FRED) plus the BTC market factor are joined onto
  every instrument's bars as causal `ctx_*` features, so both the GP generator and the
  `macro_flow` archetype can condition on *where money is flowing between assets* —
  risk-on/off, dollar liquidity, BTC→altcoin rotation — instead of one chart alone.
  Rationale, operating cadence, and how to add new flow hypotheses:
  **[docs/STRATEGY.md](docs/STRATEGY.md)**.

Everything is designed to run at **zero infrastructure cost** on GitHub Actions of a
public repo (compute, data, state, and notifications are all free). Both brokers are
pure-REST, so no always-on server is required. Git itself is the shared state bus and
audit log: every champion change, fill, and research run is a reviewable commit.

### Jobs (GitHub Actions workflows)

| Job | Workflow | Cadence | Purpose |
|-----|----------|---------|---------|
| A — data-sync | `data-sync.yml` | ~30–60 min | Incrementally fetch OHLCV (crypto + equities) → Parquet, commit `data/` |
| D — research | `research-nightly.yml` | nightly (time-boxed) | Parameter search + GP alpha discovery → validation gate → challengers |
| C — executor (paper) | `executor-paper.yml` | bar cadence | Paper trading on the same code path as live |
| B — executor (live) | `executor-live.yml` | bar cadence (gated) | Small real-money execution after paper passes |
| E — keepalive | `keepalive.yml` | weekly | Prevent 60-day cron deactivation |

## Validation pipeline (the heart)

Candidates are many, so "pick the best backtest" *always* overfits. A strategy becomes
a challenger only after passing, in order:

1. Point-in-time data, closed bars only (no look-ahead).
2. **CPCV** (Combinatorial Purged Cross-Validation) with purge + embargo.
3. **PBO** (Probability of Backtest Overfitting, CSCV).
4. **Deflated Sharpe Ratio** + minimum track-record length, honestly deflated by the
   *cumulative* number of trials ever attempted.
5. Multiple-testing control (Hansen SPA) when comparing a universe.
6. Mandatory paper incubation (shadow → canary) before any real money.

A CI self-test feeds pure noise through the gate and asserts it gets **rejected** —
proof the validation actually works.

## Safety

- API keys are **trade-only (no withdrawal permission)** and live exclusively in
  GitHub Actions Secrets. Nothing secret is ever committed.
- **Kill switch = a committed file** `state/KILL_SWITCH`. If it exists, every executor
  run flattens and halts before placing orders. Commit it from any device to stop.
- Idempotent execution: orders are the *difference* between target and current
  position, with client order IDs, so delayed/duplicate runs never double-fill.
- Hard limits: position/exposure caps, per-trade risk, daily-loss / max-drawdown halt.

## Status

Milestones M0–M8 complete: data layer, backtester, validation gate, paper executor,
parameter search, promotion/decay FSM, GP alpha discovery, meta-allocation + regime scaling,
and the canary-capped live path (OFF by default). A plain-language operator's manual (data
flow, pass/fail criteria, emergency stop) is in **[docs/MANUAL.md](docs/MANUAL.md)**.
This is **not** investment advice and ships with **no warranty**.

## Local development

```bash
uv sync            # install
uv run pytest      # run tests
uv run trade --help
```

Configuration lives in `strategies_config/` (universe, search spaces). Secrets are read
from environment variables (see `engine/core/config.py`) — locally via a `.env` you do
**not** commit; in CI via Actions Secrets.
