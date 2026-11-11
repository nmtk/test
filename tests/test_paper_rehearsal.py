"""Paper trading should rehearse the order path, not wait for promotion.

Two defects kept this engine from placing a single order in 1,200+ paper runs. Both are
about paper, which costs nothing, so neither had any reason to exist.

**The paper executor read the live book.** champions.json holds champions only, and nothing
has ever been promoted, so the paper executor had an empty book every run. But incubation
is measured by recomputing returns from stored bars: it never places an order, never pays
a spread, never rounds a position to a lot size and never has one rejected. Waiting for
promotion before exercising any of that meant the first order this engine ever sent would
go out with eight strategies at once, on the day real capital was at stake. So the paper
executor now reads a wider book — champions *plus* the gate-passers still incubating —
while the live executor still reads champions.json and nothing else.

**The no-trade band was measured against book equity.** An allocation weight is a fraction
of the book (0.06-0.34 in the live population) and a strategy's signal is a fraction of
full exposure (an ensemble averaging eight members emits 0.05, not 1.0), so the intended
exposure is routinely 1-2% of equity. Against a 2%-of-equity band, the *entire target
position* was below the threshold for opening it — not a dust rebalance being suppressed,
the position itself, on every symbol, every run. "Is this adjustment too small to bother
with?" and "is this position too small to open?" are different questions and now have
different thresholds.

With both fixed, the first order in the project's history went out: BTC/JPY 0.00155838 at
JPY 11,736,875, 1.83% of a JPY 1,000,000 book, fee JPY 21.95.
"""

from __future__ import annotations

import json

import pytest

from engine.orchestration.champions import (
    PAPER_CHAMPIONS_FILE,
    Allocation,
    Champions,
    paper_book_path,
    save_champions,
)
from engine.orchestration.orchestrator import (
    MIN_POSITION_FRACTION,
    REBALANCE_BAND,
    apply_no_trade_band,
)

BOOK = 1_000_000.0
PRICE = 10_000_000.0


def _qty(fraction: float) -> float:
    return fraction * BOOK / PRICE


# -- the no-trade band ----------------------------------------------------------------


def test_a_one_percent_position_can_be_opened():
    """The exact case that was blocked: a legitimate target below the old 2% band."""
    desired = _qty(0.0183)  # the real BTC/JPY target
    assert apply_no_trade_band(desired, 0.0, PRICE, BOOK) == desired


def test_a_dust_adjustment_to_a_held_position_is_still_ignored():
    """The property the band was written for, which the fix must not lose."""
    held = _qty(0.0183)
    nudged = _qty(0.0185)  # ~1% change in the position
    assert apply_no_trade_band(nudged, held, PRICE, BOOK) == held


def test_a_real_adjustment_to_a_held_position_goes_through():
    held = _qty(0.0183)
    doubled = _qty(0.04)
    assert apply_no_trade_band(doubled, held, PRICE, BOOK) == doubled


def test_the_band_is_relative_to_the_position_not_the_book():
    """A 10% change in a 1% position is a real trade; against book equity it was invisible."""
    held = _qty(0.01)
    bigger = _qty(0.011)
    delta_vs_book = abs(bigger - held) * PRICE / BOOK
    assert delta_vs_book < REBALANCE_BAND, "the old rule would have suppressed this"
    assert apply_no_trade_band(bigger, held, PRICE, BOOK) == bigger


def test_a_dust_position_is_not_opened():
    """Below the floor a position is worth less than the spread and below exchange minimums."""
    tiny = _qty(MIN_POSITION_FRACTION / 2.0)
    assert apply_no_trade_band(tiny, 0.0, PRICE, BOOK) == 0.0


def test_a_position_just_over_the_floor_is_opened():
    ok = _qty(MIN_POSITION_FRACTION * 2.0)
    assert apply_no_trade_band(ok, 0.0, PRICE, BOOK) == ok


def test_closing_a_position_is_never_suppressed_as_dust():
    """The anti-dust floor applies to opening only; exiting must always be possible."""
    held = _qty(MIN_POSITION_FRACTION / 2.0)  # a holding below the opening floor
    assert apply_no_trade_band(0.0, held, PRICE, BOOK) == 0.0


def test_a_short_target_is_handled_on_magnitude():
    assert apply_no_trade_band(_qty(-0.0183), 0.0, PRICE, BOOK) == _qty(-0.0183)
    assert apply_no_trade_band(_qty(-MIN_POSITION_FRACTION / 2), 0.0, PRICE, BOOK) == 0.0


@pytest.mark.parametrize("price,book", [(0.0, BOOK), (PRICE, 0.0), (-1.0, BOOK)])
def test_degenerate_inputs_hold_the_current_position(price, book):
    assert apply_no_trade_band(_qty(0.5), 1.23, price, book) == 1.23


# -- the two books --------------------------------------------------------------------


def _alloc(symbol: str) -> Allocation:
    return Allocation("tsmom", symbol, "bitbank", "1d", 0.2, {"lookback": 20})


def test_the_paper_book_is_preferred_when_present(tmp_path):
    save_champions(tmp_path / "champions.json", Champions(allocations=[], updated="a"))
    save_champions(
        tmp_path / PAPER_CHAMPIONS_FILE,
        Champions(allocations=[_alloc("BTC/JPY")], updated="b"),
    )
    assert paper_book_path(tmp_path).name == PAPER_CHAMPIONS_FILE


def test_the_live_book_is_the_fallback(tmp_path):
    """For a state directory written before the rehearsal existed, and for seed-champions."""
    save_champions(
        tmp_path / "champions.json", Champions(allocations=[_alloc("BTC/JPY")], updated="a")
    )
    assert paper_book_path(tmp_path).name == "champions.json"


def test_the_live_executor_never_reads_the_rehearsal_book():
    """Promotion must stay the only door to real capital.

    Checked against the source because the point is which *file* the live path opens, and
    an end-to-end live run needs broker credentials this suite does not have.
    """
    import inspect

    from engine.orchestration import live

    source = inspect.getsource(live)
    assert "champions.json" in source
    assert PAPER_CHAMPIONS_FILE not in source
    assert "paper_book_path" not in source


def test_promote_step_writes_both_books_and_keeps_live_to_champions(tmp_path, monkeypatch):
    """An incubating row reaches the rehearsal book; the live book stays empty."""
    import dataclasses

    from engine.core.config import load_settings
    from engine.data.store import OHLCVStore
    from engine.orchestration.population import Population, StrategyRecord
    from engine.orchestration.promotion import promote_step

    base = load_settings()
    settings = dataclasses.replace(base, state_dir=tmp_path / "state")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    # Start the clock at the newest stored bar, so the forward window is empty and the row
    # stays incubating. A start of 0 would make the whole history "forward", completing the
    # window instantly and demoting the row before it could reach either book.
    last_ts = OHLCVStore(settings.data_dir / "ohlcv").last_ts("bitbank", "BTC/JPY", "1d")
    assert last_ts is not None, "this test reads the repo's stored BTC/JPY bars"

    population = Population(settings.state_dir / "population.db")
    population.upsert(
        StrategyRecord(
            family="tsmom:BTC/JPY:1d",
            strategy="tsmom",
            symbol="BTC/JPY",
            source="bitbank",
            timeframe="1d",
            params={"lookback": 20},
            status="incubating",
            metrics={
                "ppy": 365.0,
                "n": 1200,
                "sharpe_per_bar": 0.05,
                "dsr": 0.99,
                "fee_bps": 10.0,
                "slippage_bps": 5.0,
                "research_only": False,
            },
            incubation_start_ts=last_ts,
        )
    )
    population.close()

    summary = promote_step(settings)
    assert (summary.n_promoted, summary.n_rejected) == (0, 0), "it must still be incubating"

    live_book = json.loads((settings.state_dir / "champions.json").read_text())
    paper_book = json.loads((settings.state_dir / PAPER_CHAMPIONS_FILE).read_text())
    assert live_book["allocations"] == [], "an incubating row must never reach the live book"
    families = {(a["strategy"], a["symbol"]) for a in paper_book["allocations"]}
    assert ("tsmom", "BTC/JPY") in families
    assert summary.n_paper == len(paper_book["allocations"])


def test_a_research_only_row_reaches_neither_book(tmp_path):
    """No venue can fill it, so not even the paper broker should see it."""
    import dataclasses

    from engine.core.config import load_settings
    from engine.orchestration.population import Population, StrategyRecord
    from engine.orchestration.promotion import promote_step

    base = load_settings()
    settings = dataclasses.replace(base, state_dir=tmp_path / "state")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    population = Population(settings.state_dir / "population.db")
    population.upsert(
        StrategyRecord(
            family="tsmom:DEXJPUS:1d",
            strategy="tsmom",
            symbol="DEXJPUS",
            source="fred",
            timeframe="1d",
            params={"lookback": 20},
            status="incubating",
            metrics={"ppy": 252.0, "n": 1200, "sharpe_per_bar": 0.05, "research_only": True},
            incubation_start_ts=0,
        )
    )
    population.close()

    promote_step(settings)
    for name in ("champions.json", PAPER_CHAMPIONS_FILE):
        book = json.loads((settings.state_dir / name).read_text())
        assert book["allocations"] == [], f"{name} must not carry a non-venue symbol"
