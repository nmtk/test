# WemofEA — WEMOF as an MT5 Expert Advisor

`WemofEA.mq5` mechanizes the publicly described WEMOF rules (Bollinger ±3σ touch as
*preparation*, skip anything explainable, fade the rest for a small pip target) with the
**exits the method actually specifies** — a pip take-profit and stop resting at the broker,
filling inside the bar.

It is the execution-side counterpart of this repo's `wemof_tick` archetype. The repo
researches; this trades.

## Status — not validated

The close-exit version of this method (`wemof`) was measured here on **227,047 bars** of
real USD/JPY M5 built from tick quotes, at realistic FX costs, and came back **negative**:

| | |
|---|---|
| Sharpe | **−0.16** |
| SPA p | 0.996 |
| DSR | 0.000 |

The pip-exit version this EA implements (`wemof_tick`) has **not yet returned a verdict** —
its bid/ask path was about half backfilled when this was written. So this EA is **untested,
not validated**. Demo and Strategy Tester first.

## What it does better than the research version

The Python archetype approximates the news veto with a volatility burst, because it has no
point-in-time economic calendar. MT5 ships one, so this EA calls `CalendarValueHistory`
directly and blacks out a window around scheduled events for both legs of the pair. That is
the filter the original describes, rather than a stand-in.

If the calendar comes back empty (it can in some tester configurations), `VetoNews` silently
passes everything — which is why `VetoBurst` is on by default as a second line.

## Two refusals built in

**Broker minimum stop distance.** A 1–2 pip take-profit is small enough that many brokers
reject it outright. `OnInit` reads `SYMBOL_TRADE_STOPS_LEVEL` and **refuses to start** if the
published target sits inside it, so you find out immediately instead of after a run of
rejected orders.

**Spread.** Break-even win rate is `SL / (SL + TP)` — 71% at TP 2 / SL 5 — and the spread
comes straight off the target, pushing that to ~76% at 0.3 pips. When the spread approaches
the target the expectancy is negative by arithmetic, whatever the chart shows. `MaxSpreadPips`
is a hard skip, and a spread at or above the target is logged and refused.

This is also why the instrument matters. On a BTC/USD CFD quoting a 10 USD spread at ~81,500
(1.23 bps), the published 1–2 pip target converts to roughly 0.7–1.3 bps — **smaller than the
spread**. The method cannot be run there as written.

## Setup

MT5 **mobile apps cannot run EAs** — no MQL5 runtime, no tester. You need the desktop
terminal, normally on a VPS so it stays on:

```
Windows VPS (always on)
  └ MT5 desktop + WemofEA        ← places the orders
        ↕ same account
  MT5 mobile app                 ← monitoring only
```

1. Copy `WemofEA.mq5` to `MQL5/Experts/` in the terminal's data folder
   (File → Open Data Folder), then compile it in MetaEditor (F7).
2. Open a **USD/JPY M5** chart and attach the EA. Allow algorithmic trading.
3. Defaults match the published method: `BBPeriod 20`, `BBSigma 3.0`, `TakeProfitPips 1.5`,
   `StopLossPips 5.0`, `MaxHoldBars 6`, three vetoes on, trend veto off.
4. Test with **"Every tick based on real ticks"** so the tester uses the broker's own tick
   data and spread. A model coarser than that cannot resolve a 1.5 pip target honestly —
   the same reason the research version refused to simulate these exits from OHLCV.

## Inputs worth understanding

| Input | Why it matters |
|---|---|
| `BBSigma` | 3.0, not 2.0. At 2σ the touch frequency is ~15× higher and the "rare outlier" premise is gone. The EA warns below 2.5. |
| `TakeProfitPips` / `StopLossPips` | Sets the break-even win rate. Printed to the log on init. |
| `MaxSpreadPips` | A refusal, not a preference. See above. |
| `VetoNews` | The real calendar. The core of the method. |
| `VetoBurst` | Secondary, and the only news-like filter that survives a tester without calendar data. |
| `MaxHoldBars` | A scalp does not sit in a losing position hoping. |

## Faithfulness notes

Reproduced from the published, verbally described rules only — no proprietary indicator code.
Entries are evaluated on **closed bars**: a forming bar's high and low still move, and the
bands shift with each new bar, so a touch seen mid-bar may not be there when the bar
completes.
