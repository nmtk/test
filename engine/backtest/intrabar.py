"""Bracket-exit backtest on the intrabar bid/ask path.

``run_backtest`` earns close-to-close returns on a weight series, which is the right
model for anything that decides and exits on bar closes. A scalp with a take-profit and a
stop does neither: both orders rest at the broker and fill *inside* the bar, on the
quote's own side. This module replays such trades against the one-minute bid/ask path
(``engine/data/intrabar.py``) and turns them back into per-bar returns, so the unchanged
validation gate (CPCV, PBO, DSR, SPA) can judge them like any other strategy.

Fill rules, written for a long (a short mirrors them on the ask):

* **entry** — market, at the ask of the first quote at or after the deciding bar's
  close. If no quote arrives before the next bar would have closed (a data hole, the
  weekend), the signal is dropped: a market order into a closed market is rejected.
* **take-profit** — a sell limit at ``entry + tp``. Fills in the first minute whose bid
  high reaches it, at the level and never better, even when the minute opens beyond it.
* **stop** — a sell stop at ``entry - sl``. A minute that *opens* through it fills at
  that open (the gap is real and is paid in full). Otherwise, in the first minute whose
  bid low reaches it, it fills at ``max(level - bid_low_step, bid_low)``. Every earlier
  tick of that minute was above the level, so the tick that triggered the stop made a
  new low for the minute, and no new low in it undercut the one before by more than
  ``bid_low_step`` — this price is never better than where the stop really filled.
* **both touched in one minute** — the order inside the minute is unknown, so the stop
  wins, at the bound above.
* **time exit** — market, at the bid of the first quote at or after ``max_hold`` bars
  past the deciding close. **End of data** inside a trade closes it at the last bid.

Together these make every trade's P&L at most what the same trade earns replayed tick by
tick — the simulator can lose edge to its one-minute resolution but cannot invent any.
``tests/test_intrabar_sim.py`` checks that against a tick-level reference.

Return attribution is mark-to-market on the bar grid: a long is valued at the bid close
of each bar it spans (a short at the ask close), so the spread is booked the moment a
position opens, and the per-bar returns compound to the trade's own return. A trade
decided on bar ``t``'s close earns nothing before bar ``t + 1`` — the same no-look-ahead
rule ``run_backtest`` enforces with its one-bar shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.backtest.engine import BacktestResult
from engine.backtest.metrics import compute_stats
from engine.data.intrabar import IntrabarPath

TRADE_COLUMNS = (
    "signal_bar",
    "side",
    "entry_ts",
    "entry_px",
    "exit_ts",
    "exit_px",
    "reason",
    "ambiguous",
    "gross_return",
)


@dataclass(slots=True)
class IntrabarBacktest:
    result: BacktestResult
    trades: pd.DataFrame
    # Signals that could not be filled because no quote followed them in time.
    n_unfilled: int = 0

    @property
    def ambiguous_share(self) -> float:
        """Fraction of trades whose exit minute touched both levels (resolved to the stop)."""
        if len(self.trades) == 0:
            return 0.0
        return float(self.trades["ambiguous"].mean())


def _exit_scan(path: IntrabarPath, k: int, j_end: int, side: int, upper: float, lower: float):
    """First exit inside sub-bars ``[k, j_end)``: ``(index, price, reason, ambiguous)``.

    For a long ``upper`` is the take-profit and ``lower`` the stop, both read on the bid;
    for a short ``lower`` is the take-profit and ``upper`` the stop, read on the ask.
    """
    if j_end <= k:
        return None
    window = slice(k, j_end)
    if side > 0:
        o, h, lo = path.bid_open[window], path.bid_high[window], path.bid_low[window]
        gap_stop, gap_take = o <= lower, o >= upper
        hit_stop, hit_take = lo <= lower, h >= upper
    else:
        o, h, lo = path.ask_open[window], path.ask_high[window], path.ask_low[window]
        gap_stop, gap_take = o >= upper, o <= lower
        hit_stop, hit_take = h >= upper, lo <= lower

    event = np.flatnonzero(hit_stop | hit_take)
    if len(event) == 0:
        return None
    f = int(event[0])
    j = k + f
    if gap_stop[f]:
        return j, float(o[f]), "stop_gap", False
    if gap_take[f]:
        return j, float(upper if side > 0 else lower), "take_profit", False
    if hit_stop[f]:
        if side > 0:
            px = max(lower - float(path.bid_low_step[j]), float(path.bid_low[j]))
        else:
            px = min(upper + float(path.ask_high_step[j]), float(path.ask_high[j]))
        return j, px, "stop", bool(hit_take[f])
    return j, float(upper if side > 0 else lower), "take_profit", False


def simulate_brackets(
    bar_ts: np.ndarray,
    bar_ms: int,
    path: IntrabarPath,
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    *,
    take_profit: float,
    stop_loss: float,
    max_hold: int,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    periods_per_year: float = 365.0,
    index: pd.Index | None = None,
) -> IntrabarBacktest:
    """Replay one position at a time with a take-profit/stop bracket and a time limit.

    ``long_signal[i]`` / ``short_signal[i]`` mean "enter at bar ``i``'s close"; a signal
    is ignored while a position is open. ``take_profit`` and ``stop_loss`` are distances
    in price units from the fill price. ``fee_bps + slippage_bps`` is charged on every
    fill, on top of the spread the bid/ask path already contains. ``index`` labels the
    returned per-bar series (the bars' own index).
    """
    bar_ts = np.asarray(bar_ts, dtype="int64")
    long_signal = np.asarray(long_signal, dtype=bool)
    short_signal = np.asarray(short_signal, dtype=bool)
    n = len(bar_ts)
    net = np.zeros(n)
    applied = np.zeros(n)
    turnover = np.zeros(n)
    trades: list[tuple] = []
    unfilled = 0

    pts = path.ts
    m = len(pts)
    cost = (fee_bps + slippage_bps) / 10_000.0
    # Last sub-bar finished by each bar's close: where a position is marked at that close.
    mark_idx = np.searchsorted(pts + path.step_ms, bar_ts + bar_ms, side="right") - 1

    free_from = np.iinfo(np.int64).min
    for i in np.flatnonzero(long_signal | short_signal):
        decision = int(bar_ts[i]) + bar_ms
        if decision < free_from:
            continue
        if i + 1 >= n:
            break  # nothing left to earn the trade in
        side = 1 if long_signal[i] else -1

        k = int(np.searchsorted(pts, decision, side="left"))
        if k >= m or pts[k] >= decision + bar_ms:
            unfilled += 1
            continue
        entry = float(path.ask_open[k] if side > 0 else path.bid_open[k])
        if side > 0:
            upper, lower = entry + take_profit, entry - stop_loss
        else:
            upper, lower = entry + stop_loss, entry - take_profit

        deadline = decision + int(max_hold) * bar_ms
        j_end = int(np.searchsorted(pts, deadline, side="left"))
        hit = _exit_scan(path, k, j_end, side, upper, lower)
        if hit is not None:
            j_exit, exit_px, reason, ambiguous = hit
            exit_ts = int(pts[j_exit])
            free_from = exit_ts + path.step_ms
        elif j_end < m:
            j_exit = j_end
            exit_px = float(path.bid_open[j_exit] if side > 0 else path.ask_open[j_exit])
            reason, ambiguous = "time", False
            exit_ts = int(pts[j_exit])
            free_from = exit_ts + 1
        else:
            j_exit = m - 1
            exit_px = float(path.bid_close[j_exit] if side > 0 else path.ask_close[j_exit])
            reason, ambiguous = "end_of_data", False
            exit_ts = int(pts[j_exit])
            free_from = np.iinfo(np.int64).max

        e = max(int(np.searchsorted(bar_ts, int(pts[k]), side="right")) - 1, i + 1)
        x = min(max(int(np.searchsorted(bar_ts, exit_ts, side="right")) - 1, e), n - 1)
        closes = path.bid_close if side > 0 else path.ask_close
        prev = entry
        for b in range(e, x):
            mi = int(mark_idx[b])
            mark = float(closes[mi]) if k <= mi < j_exit else prev
            net[b] += side * (mark / prev - 1.0)
            prev = mark
        net[x] += side * (exit_px / prev - 1.0)
        net[e] -= cost
        net[x] -= cost
        turnover[e] += 1.0
        turnover[x] += 1.0
        applied[e : x + 1] = side

        trades.append(
            (
                int(i),
                side,
                int(pts[k]),
                entry,
                exit_ts,
                exit_px,
                reason,
                ambiguous,
                side * (exit_px / entry - 1.0),
            )
        )
        if free_from == np.iinfo(np.int64).max:
            break

    returns = pd.Series(net, index=index)
    equity = (1.0 + returns).cumprod()
    applied_s = pd.Series(applied, index=index)
    turnover_s = pd.Series(turnover, index=index)
    stats = compute_stats(returns, equity, applied_s, turnover_s, periods_per_year)
    stats["n_trades"] = float(len(trades))
    result = BacktestResult(
        equity=equity,
        returns=returns,
        applied_weights=applied_s,
        turnover=turnover_s,
        stats=stats,
    )
    frame = pd.DataFrame(trades, columns=list(TRADE_COLUMNS))
    return IntrabarBacktest(result=result, trades=frame, n_unfilled=unfilled)
