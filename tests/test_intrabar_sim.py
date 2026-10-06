"""Bracket-exit simulator: fill rules, attribution, causality — and the guarantee.

The guarantee is the point of the module: replayed on one-minute bid/ask bars, a trade
may lose edge to the coarse resolution but must never *earn* more than the same trade
replayed tick by tick. ``test_never_beats_the_tick_exact_replay`` checks that against an
independent tick-level reference on random streams full of gaps, jumps, wide quotes and
minutes that touch both levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.backtest.intrabar import simulate_brackets
from engine.data.dukascopy import ticks_to_ohlcv
from engine.data.intrabar import IntrabarPath, ticks_to_path

MIN = 60_000
BAR = 5 * MIN
T0 = 1_704_067_200_000  # a 5m boundary


def _path(ticks: list[tuple[int, float, float]]) -> IntrabarPath:
    frame = pd.DataFrame(ticks, columns=["ts", "bid", "ask"])
    return IntrabarPath.from_frame(ticks_to_path(frame, "USD/JPY"), "USD/JPY")


def _minutes(rows, start=T0, spread=0.002):
    """Four ticks per minute in the order open, high, low, close."""
    ticks = []
    for i, row in enumerate(rows):
        t = start + i * MIN
        for dt, bid in zip((0, 10_000, 20_000, 30_000), row, strict=True):
            ticks.append((t + dt, bid, bid + spread))
    return ticks


FLAT = (150.000, 150.000, 150.000, 150.000)


def _run(path, n_bars=6, long_at=(0,), short_at=(), tp=0.01, sl=0.03, hold=2, **kw):
    bar_ts = T0 + np.arange(n_bars) * BAR
    long_sig = np.zeros(n_bars, dtype=bool)
    short_sig = np.zeros(n_bars, dtype=bool)
    long_sig[list(long_at)] = True
    short_sig[list(short_at)] = True
    return simulate_brackets(
        bar_ts, BAR, path, long_sig, short_sig, take_profit=tp, stop_loss=sl, max_hold=hold, **kw
    )


# -- fill rules -------------------------------------------------------------------------------


def test_take_profit_fills_at_the_level_in_the_bar_after_the_signal():
    rows = [FLAT] * 6 + [(150.000, 150.020, 149.995, 150.010)] + [FLAT] * 23
    sim = _run(_path(_minutes(rows)))
    (trade,) = sim.trades.itertuples()
    assert trade.reason == "take_profit"
    assert trade.entry_px == pytest.approx(150.002), "a long pays the ask"
    assert trade.exit_px == pytest.approx(150.012)
    returns = sim.result.returns.to_numpy()
    assert returns[0] == 0.0, "nothing is earned on the bar whose close decided the trade"
    assert returns[1] == pytest.approx(150.012 / 150.002 - 1)


def test_stop_fill_is_bounded_by_the_largest_jump_and_never_better():
    # Entry 150.002 (ask), stop 149.972. Bid ticks: 150.000 -> 149.970 (the true fill)
    # -> 149.900 inside the same minute.
    ticks = _minutes([FLAT] * 6, start=T0) + [
        (T0 + 6 * MIN, 150.000, 150.002),
        (T0 + 6 * MIN + 10_000, 149.970, 149.972),
        (T0 + 6 * MIN + 20_000, 149.900, 149.902),
    ]
    ticks += _minutes([FLAT] * 23, start=T0 + 7 * MIN)
    (trade,) = _run(_path(ticks)).trades.itertuples()
    assert trade.reason == "stop"
    assert 149.900 <= trade.exit_px <= 149.970


def test_stop_fill_equals_the_tick_fill_when_the_jump_bound_is_tight():
    ticks = _minutes([FLAT] * 6) + [
        (T0 + 6 * MIN, 150.000, 150.002),
        (T0 + 6 * MIN + 10_000, 149.985, 149.987),
        (T0 + 6 * MIN + 20_000, 149.968, 149.970),
    ]
    ticks += _minutes([FLAT] * 23, start=T0 + 7 * MIN)
    (trade,) = _run(_path(ticks)).trades.itertuples()
    assert trade.exit_px == pytest.approx(149.968)


def test_a_minute_that_opens_through_the_stop_pays_the_whole_gap():
    rows = [FLAT] * 6 + [(149.900, 149.910, 149.890, 149.900)] + [FLAT] * 23
    (trade,) = _run(_path(_minutes(rows))).trades.itertuples()
    assert trade.reason == "stop_gap"
    assert trade.exit_px == pytest.approx(149.900)


def test_a_minute_that_opens_beyond_the_target_fills_at_the_target_not_better():
    rows = [FLAT] * 6 + [(150.100, 150.110, 150.090, 150.100)] + [FLAT] * 23
    (trade,) = _run(_path(_minutes(rows))).trades.itertuples()
    assert trade.reason == "take_profit"
    assert trade.exit_px == pytest.approx(150.012)


def test_touching_both_levels_in_one_minute_resolves_to_the_stop():
    # The high comes first in reality — the simulator cannot know that, so it must not
    # assume it.
    rows = [FLAT] * 6 + [(150.000, 150.020, 149.960, 150.000)] + [FLAT] * 23
    sim = _run(_path(_minutes(rows)))
    (trade,) = sim.trades.itertuples()
    assert trade.reason == "stop"
    assert bool(trade.ambiguous) is True
    assert trade.exit_px < 150.002
    assert sim.ambiguous_share == 1.0


def test_time_exit_is_the_first_quote_after_the_deadline():
    # Quotes stop for 12 minutes straddling the deadline (decision + 2 bars = minute 15).
    ticks = _minutes([FLAT] * 10) + _minutes([(150.004,) * 4] * 20, start=T0 + 22 * MIN)
    (trade,) = _run(_path(ticks), n_bars=10).trades.itertuples()
    assert trade.reason == "time"
    assert trade.exit_ts == T0 + 22 * MIN
    assert trade.exit_px == pytest.approx(150.004), "a long sells at the bid"


def test_a_trade_open_when_the_data_ends_is_closed_at_the_last_bid():
    rows = [FLAT] * 6 + [(150.000, 150.005, 149.995, 150.003)] * 3
    sim = _run(_path(_minutes(rows)), n_bars=2, hold=12)
    (trade,) = sim.trades.itertuples()
    assert trade.reason == "end_of_data"
    assert trade.exit_px == pytest.approx(150.003)


def test_a_short_mirrors_everything_on_the_ask():
    # Entry at the bid 150.000; target 149.990 and stop 150.030, both read on the ask.
    rows = [FLAT] * 6 + [(150.000, 150.001, 149.985, 149.990)] + [FLAT] * 23
    sim = _run(_path(_minutes(rows, spread=0.002)), long_at=(), short_at=(0,))
    (trade,) = sim.trades.itertuples()
    assert trade.side == -1
    assert trade.entry_px == pytest.approx(150.000), "a short sells the bid"
    assert trade.reason == "take_profit"
    assert trade.exit_px == pytest.approx(149.990)
    assert trade.gross_return == pytest.approx(-(149.990 / 150.000 - 1))


def test_an_unfillable_signal_is_dropped_not_filled_late():
    # No quote at all in the bar after the signal (minutes 5-9), quotes resume at 10.
    ticks = _minutes([FLAT] * 5) + _minutes([FLAT] * 20, start=T0 + 10 * MIN)
    sim = _run(_path(ticks))
    assert len(sim.trades) == 0
    assert sim.n_unfilled == 1


def test_signals_are_ignored_while_a_position_is_open():
    sim = _run(_path(_minutes([FLAT] * 60)), n_bars=12, long_at=(0, 1, 2, 3), hold=2)
    # 0 -> time exit at the open of bar 3 (decision 5m + 2 bars); 1 and 2 are inside the
    # trade; 3 closes after the exit, so it trades.
    assert list(sim.trades["signal_bar"]) == [0, 3]


# -- returns ------------------------------------------------------------------------------------


def test_spread_is_the_whole_cost_of_a_trade_that_goes_nowhere():
    sim = _run(_path(_minutes([FLAT] * 30)))
    (trade,) = sim.trades.itertuples()
    assert trade.gross_return == pytest.approx(150.000 / 150.002 - 1)
    assert sim.result.returns.sum() == pytest.approx(trade.gross_return)


def test_fees_are_charged_on_both_fills():
    path = _path(_minutes([FLAT] * 30))
    gross = _run(path).result.returns.sum()
    net = _run(path, fee_bps=1.0, slippage_bps=0.5).result.returns.sum()
    assert gross - net == pytest.approx(2 * 1.5 / 10_000)


def test_per_bar_returns_compound_to_the_trades(make_ticks):
    ticks = make_ticks(hours=12, seed=5)
    bars = ticks_to_ohlcv(ticks, "5m")
    path = IntrabarPath.from_frame(ticks_to_path(ticks, "USD/JPY"), "USD/JPY")
    rng = np.random.default_rng(1)
    long_sig = rng.random(len(bars)) < 0.3
    sim = simulate_brackets(
        bars["ts"].to_numpy(),
        BAR,
        path,
        long_sig,
        np.zeros(len(bars), dtype=bool),
        take_profit=0.015,
        stop_loss=0.05,
        max_hold=4,
    )
    assert len(sim.trades) > 20
    assert (1 + sim.result.returns).prod() == pytest.approx(
        (1 + sim.trades["gross_return"]).prod(), rel=1e-12
    )


def test_returns_before_a_moment_ignore_every_tick_after_it(make_ticks):
    ticks = make_ticks(hours=12, seed=7)
    bars = ticks_to_ohlcv(ticks, "5m")
    cut = int(bars["ts"].iloc[len(bars) // 2])
    rng = np.random.default_rng(2)
    long_sig = rng.random(len(bars)) < 0.15
    short_sig = ~long_sig & (rng.random(len(bars)) < 0.15)

    def run(stream):
        path = IntrabarPath.from_frame(ticks_to_path(stream, "USD/JPY"), "USD/JPY")
        return simulate_brackets(
            bars["ts"].to_numpy(),
            BAR,
            path,
            long_sig,
            short_sig,
            take_profit=0.015,
            stop_loss=0.05,
            max_hold=6,
        ).result.returns.to_numpy()

    altered = ticks.copy()
    later = altered["ts"] >= cut
    shock = np.random.default_rng(3).normal(0, 0.05, int(later.sum()))
    altered.loc[later, "bid"] += shock
    altered.loc[later, "ask"] += shock

    before = (bars["ts"] + BAR <= cut).to_numpy()
    np.testing.assert_array_equal(run(ticks)[before], run(altered)[before])


# -- the guarantee --------------------------------------------------------------------------------


def _tick_reference(ticks, bar_ts, long_sig, short_sig, tp, sl, hold):
    """Tick-by-tick replay under the same entry, time-exit and one-position rules."""
    ts = ticks["ts"].to_numpy()
    bid = ticks["bid"].to_numpy()
    ask = ticks["ask"].to_numpy()
    n = len(bar_ts)
    out = []
    free_from = -np.inf
    for i in np.flatnonzero(long_sig | short_sig):
        decision = bar_ts[i] + BAR
        if decision < free_from:
            continue
        if i + 1 >= n:
            break
        side = 1 if long_sig[i] else -1
        k = int(np.searchsorted(ts, decision, side="left"))
        if k >= len(ts) or ts[k] >= decision + BAR:
            continue
        entry = ask[k] if side > 0 else bid[k]
        upper, lower = (entry + tp, entry - sl) if side > 0 else (entry + sl, entry - tp)
        deadline = decision + hold * BAR
        t, exit_px = k, None
        while t < len(ts) and ts[t] < deadline:
            if side > 0 and bid[t] <= lower:
                exit_px = bid[t]
            elif side > 0 and bid[t] >= upper:
                exit_px = upper
            elif side < 0 and ask[t] >= upper:
                exit_px = ask[t]
            elif side < 0 and ask[t] <= lower:
                exit_px = lower
            if exit_px is not None:
                break
            t += 1
        if exit_px is not None:
            free_from = (ts[t] // MIN + 1) * MIN
        elif t < len(ts):
            exit_px = bid[t] if side > 0 else ask[t]
            free_from = ts[t] + 1
        else:
            t = len(ts) - 1
            exit_px = bid[t] if side > 0 else ask[t]
            free_from = np.inf
        out.append((int(i), side, entry, int(ts[t] // MIN * MIN), side * (exit_px / entry - 1)))
        if free_from == np.inf:
            break
    return out


def test_never_beats_the_tick_exact_replay(make_ticks):
    shortfalls: list[float] = []
    n_ambiguous = 0
    for seed in (11, 12, 13, 14):
        ticks = make_ticks(hours=48, seed=seed, jump_rate=0.004, wide_rate=0.004)
        bars = ticks_to_ohlcv(ticks, "5m")
        path = IntrabarPath.from_frame(ticks_to_path(ticks, "USD/JPY"), "USD/JPY")
        rng = np.random.default_rng(seed)
        long_sig = rng.random(len(bars)) < 0.2
        short_sig = ~long_sig & (rng.random(len(bars)) < 0.2)
        tp, sl, hold = 0.012, 0.035, 3

        sim = simulate_brackets(
            bars["ts"].to_numpy(),
            BAR,
            path,
            long_sig,
            short_sig,
            take_profit=tp,
            stop_loss=sl,
            max_hold=hold,
        )
        ref = _tick_reference(ticks, bars["ts"].to_numpy(), long_sig, short_sig, tp, sl, hold)

        assert len(sim.trades) == len(ref) > 100
        for trade, (bar, side, entry, exit_minute, ref_return) in zip(
            sim.trades.itertuples(), ref, strict=True
        ):
            assert (trade.signal_bar, trade.side) == (bar, side)
            assert trade.entry_px == pytest.approx(entry, abs=1e-9)
            assert trade.exit_ts == exit_minute, "the exit happens in the same minute"
            assert trade.gross_return <= ref_return + 1e-12
            if trade.reason != "stop":
                # Only a stop's fill is bounded rather than known; every other exit is exact.
                assert trade.gross_return == pytest.approx(ref_return, abs=1e-12)
            shortfalls.append(ref_return - trade.gross_return)
        n_ambiguous += int(sim.trades["ambiguous"].sum())

    # Not vacuous: the streams do produce minutes the coarse resolution pays for ...
    assert max(shortfalls) > 0
    assert n_ambiguous > 0
    # ... and what it pays is small next to a 1.2-pip target (returns at a ~150 price).
    pip_return = 0.01 / 150.0
    assert np.mean(shortfalls) < 0.25 * pip_return
