"""WEMOF with the original's exits: a pip take-profit and stop that fill inside the bar.

``wemof`` exits on bar closes because 5m OHLCV cannot say which of a 1-pip target and a
5-pip stop a bar touched first, and inventing that order would invent fills. The
intrabar bid/ask path (``engine/data/intrabar.py``) answers that question to the minute,
so this archetype trades the **same entries** — the same Bollinger overshoot and the same
four "is it explainable?" vetoes, through shared code — with the exits the method
actually describes:

* a take-profit ``tp_pips`` and a stop ``sl_pips`` away from the fill, resting at the
  broker and filling inside the bar on the quote's own side;
* a time limit of ``max_hold`` bars, because the method is a scalp.

The spread is not an assumption here. Entries pay the ask (long) or the bid (short) of
the first quote after the signal, and exits are read on the opposite side, all from the
path. The fill rules — including how a minute that touches both levels is resolved
against the trade — are in ``engine/backtest/intrabar.py``.

Research-only, permanently for now: the paper and live executors set one target
position per bar close, from a cron that fires minutes late, and have no way to place a
bracket. Results this archetype earns in the gate describe a trade nothing here can
execute, so research records it as research-only and the promotion FSM never allocates
to it. Executing it would need an always-on process (or broker-side OCO brackets placed
the instant a bar closes) — a different system, not a config change.
"""

from __future__ import annotations

import pandas as pd

from engine.backtest.engine import BacktestResult
from engine.backtest.intrabar import IntrabarBacktest, simulate_brackets
from engine.data.intrabar import intrabar_of
from engine.strategies.base import FloatParam, IntParam, register
from engine.strategies.wemof import WemofReversal


class IntrabarRequired(RuntimeError):
    """Raised when an intrabar strategy is asked for something bars alone cannot answer."""


@register
class WemofTick(WemofReversal):
    """WEMOF entries, bracketed by a pip take-profit/stop, simulated on the bid/ask path."""

    name = "wemof_tick"
    # Research reads these: run only where a path exists, only on the method's own
    # timeframe, and inherit the close-exit family's trial count (same entry search).
    requires_intrabar = True
    intrabar_timeframes = ("5m",)
    parent_family = "wemof"

    param_space = {
        "bb_window": IntParam(10, 40),
        "bb_sigma": FloatParam(1.8, 3.5),
        # The published method: ~1-2 pips of profit against a ~3-8 pip stop.
        "tp_pips": FloatParam(0.5, 3.0),
        "sl_pips": FloatParam(2.0, 10.0),
        "max_hold": IntParam(1, 12),
        "veto_structure": [0, 1],
        "veto_level": [0, 1],
        "veto_news": [0, 1],
        "veto_trend": [0, 1],
        "structure_window": IntParam(10, 120),
        "level_decades": IntParam(2, 4),
        "level_tolerance": FloatParam(0.02, 0.25),
        "news_atr_mult": FloatParam(1.2, 4.0),
        "trend_window": IntParam(20, 200),
    }

    def __init__(
        self,
        bb_window: int = 20,
        bb_sigma: float = 3.0,
        tp_pips: float = 1.5,
        sl_pips: float = 5.0,
        max_hold: int = 6,
        veto_structure: int = 1,
        veto_level: int = 1,
        veto_news: int = 1,
        veto_trend: int = 0,
        structure_window: int = 40,
        level_decades: int = 3,
        level_tolerance: float = 0.08,
        news_atr_mult: float = 2.0,
        trend_window: int = 100,
    ) -> None:
        super().__init__(
            bb_window=bb_window,
            bb_sigma=bb_sigma,
            max_hold=max_hold,
            veto_structure=veto_structure,
            veto_level=veto_level,
            veto_news=veto_news,
            veto_trend=veto_trend,
            structure_window=structure_window,
            level_decades=level_decades,
            level_tolerance=level_tolerance,
            news_atr_mult=news_atr_mult,
            trend_window=trend_window,
        )
        self.tp_pips = float(tp_pips)
        self.sl_pips = float(sl_pips)

    def weights(self, bars: pd.DataFrame) -> pd.Series:
        raise IntrabarRequired(
            "wemof_tick fills inside bars, so its returns are not a close-to-close weight "
            "series; use backtest()/simulate() on bars carrying an intrabar path"
        )

    def simulate(
        self,
        bars: pd.DataFrame,
        *,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        periods_per_year: float = 365.0,
    ) -> IntrabarBacktest:
        attached = intrabar_of(bars)
        if attached is None:
            raise IntrabarRequired(
                "no intrabar path on these bars — load one with "
                "engine.data.intrabar.prepare_intrabar_bars (run `trade sync-data` first)"
            )
        path, bar_ms = attached
        long_signal, short_signal, _ = self.entry_signals(bars)
        return simulate_brackets(
            bars["ts"].to_numpy(dtype="int64"),
            bar_ms,
            path,
            long_signal,
            short_signal,
            take_profit=self.tp_pips * path.pip,
            stop_loss=self.sl_pips * path.pip,
            max_hold=self.max_hold,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            periods_per_year=periods_per_year,
            index=bars.index,
        )

    def backtest(
        self,
        bars: pd.DataFrame,
        *,
        fee_bps: float = 0.0,
        slippage_bps: float = 0.0,
        periods_per_year: float = 365.0,
    ) -> BacktestResult:
        return self.simulate(
            bars, fee_bps=fee_bps, slippage_bps=slippage_bps, periods_per_year=periods_per_year
        ).result
