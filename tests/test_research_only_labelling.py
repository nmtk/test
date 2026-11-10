"""A family discovered on a non-venue series must be *labelled* as such, not just guarded.

`promote_step` has always refused to promote a record whose metrics say
``research_only``, and that refusal was tested. What was never tested is whether research
actually sets the flag — and the ensemble branch of ``run_research`` did not. It omitted
``research_only`` entirely and took the parameter's ``False`` default, so every pool built
on a research-only series was mislabelled.

The consequence was not cosmetic. ``ensemble:DEXJPUS:1d`` cleared the gate on 2026-11-08
and entered ``incubating`` with ``research_only=False``, which made it eligible for
promotion to champion. DEXJPUS is a FRED series — 55 years of USD/JPY, which is what makes
the macro archetypes judgeable, but not an instrument any broker carries. A promotion would
have handed the executor a symbol that is not in the universe, and ``_resolve_instrument``
fabricates one for an unknown symbol rather than failing.

The lesson is the shape of the gap, not the typo: the guard was tested with a correctly
labelled record, so it proved nothing about the labelling. These tests check the property
over *every* family research writes, so a fourth generator added later cannot reintroduce
it by forgetting one argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.orchestration.population import Population
from engine.search.research import _record_for, run_research
from engine.validation.gate import GateThresholds

LENIENT = GateThresholds(
    dsr_min=-1.0, pbo_max=1.0, spa_max=1.0, min_oos_sharpe=-1e9, min_pct_positive=0.0
)


def _report(passed: bool = True):
    import types

    return types.SimpleNamespace(
        passed=passed,
        best_params={"lookback": 20},
        dsr=0.99,
        pbo=0.05,
        spa_p=0.01,
        cpcv_median=1.5,
        sharpe_annual=2.0,
        sharpe_per_bar=0.1,
        mean_per_bar=0.001,
        std_per_bar=0.01,
        skew=0.0,
        kurtosis=3.0,
        max_drawdown=-0.05,
        n_bars=1200,
    )


def test_record_for_refuses_to_guess_the_flag():
    """It used to default to False, so forgetting it produced the unsafe value silently."""
    with pytest.raises(TypeError, match="research_only"):
        _record_for(
            _report(),
            family="ensemble:DEXJPUS:1d",
            strat_name="ensemble",
            symbol="DEXJPUS",
            source="fred",
            tf="1d",
            ppy=252.0,
            last_ts=0,
            existing=None,
        )


@pytest.mark.parametrize("flag", [True, False])
def test_the_flag_reaches_the_record(flag):
    rec = _record_for(
        _report(),
        family="ensemble:DEXJPUS:1d",
        strat_name="ensemble",
        symbol="DEXJPUS",
        source="fred",
        tf="1d",
        ppy=252.0,
        last_ts=0,
        existing=None,
        research_only=flag,
    )
    assert rec.metrics["research_only"] is flag


def _write_research_series(settings) -> None:
    """A long, well-behaved series for a symbol that is NOT in the tradeable universe."""
    n = 900
    rng = np.random.default_rng(0)
    close = 150.0 * np.cumprod(1.0 + rng.normal(0.0008, 0.01, n))
    frame = pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * 86_400_000 + 1_600_000_000_000,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
        }
    )
    path = settings.data_dir / "ohlcv" / "fred" / "DEXJPUS" / "1d.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_every_family_written_for_a_research_series_is_labelled(paper_settings):
    """The general property, over whatever generators happen to be enabled.

    This is the test that would have caught the ensemble omission: it does not name the
    call sites, it asserts the invariant across all of them.
    """
    _write_research_series(paper_settings)
    run_research(
        paper_settings,
        strategies=("tsmom", "donchian"),
        symbols=["DEXJPUS"],
        timeframes=["1d"],
        trials=4,
        thresholds=LENIENT,
        min_bars=300,
        include_ensemble=True,
    )

    population = Population(paper_settings.state_dir / "population.db")
    try:
        rows = [r for r in population.list() if r["symbol"] == "DEXJPUS"]
    finally:
        population.close()

    assert rows, "research wrote nothing for the research-only series"
    assert {r["family"].split(":")[0] for r in rows} >= {"ensemble"}, (
        "the ensemble pool must be among the families checked — it is the one that was wrong"
    )
    mislabelled = [r["family"] for r in rows if not r["metrics"].get("research_only")]
    assert not mislabelled, f"promotable rows on a non-venue series: {mislabelled}"


def test_a_tradeable_series_is_not_labelled_research_only(paper_settings):
    """The flag must discriminate, not be pinned on."""
    run_research(
        paper_settings,
        strategies=("tsmom",),
        symbols=["BTC/JPY"],
        timeframes=["1h"],
        trials=4,
        thresholds=LENIENT,
        min_bars=150,
    )
    population = Population(paper_settings.state_dir / "population.db")
    try:
        rows = [r for r in population.list() if r["symbol"] == "BTC/JPY"]
    finally:
        population.close()
    assert rows
    assert all(r["metrics"].get("research_only") is False for r in rows)


# -- status, not just the flag --------------------------------------------------------


def test_a_research_only_pass_stays_a_candidate():
    """`incubating` is defined by a forward test this row will never be given.

    promote_step skips research-only records before computing forward returns, so nothing
    would ever move such a row out of `incubating` again. It would sit there permanently
    labelled "awaiting a verdict" while awaiting nothing, and it would inflate the
    incubating count — the number read as "strategies on their way to trading" — with a
    row that can never trade. ensemble:DEXJPUS:1d did exactly that for three nights.
    """
    rec = _record_for(
        _report(passed=True),
        family="ensemble:DEXJPUS:1d",
        strat_name="ensemble",
        symbol="DEXJPUS",
        source="fred",
        tf="1d",
        ppy=252.0,
        last_ts=1_000,
        existing=None,
        research_only=True,
    )
    assert rec.status == "candidate"
    assert rec.incubation_start_ts is None, "no clock, because no verdict is coming"
    assert rec.metrics["passed"] is True, "the pass itself must not be lost"
    assert rec.metrics["research_only"] is True


def test_a_tradeable_pass_still_incubates():
    """The contrast that makes the previous test meaningful."""
    rec = _record_for(
        _report(passed=True),
        family="tsmom:BTC/JPY:1d",
        strat_name="tsmom",
        symbol="BTC/JPY",
        source="bitbank",
        tf="1d",
        ppy=365.0,
        last_ts=1_000,
        existing=None,
        research_only=False,
    )
    assert rec.status == "incubating" and rec.incubation_start_ts == 1_000


def test_a_research_only_pass_is_not_mistaken_for_a_reject():
    """shadow.py tracks *rejects*; a passing row is not one, whatever its status."""
    from engine.orchestration.shadow import _rejected

    passing = {"status": "candidate", "metrics": {"passed": True, "research_only": True}}
    failing = {"status": "candidate", "metrics": {"passed": False, "research_only": True}}
    assert _rejected(passing) is False
    assert _rejected(failing) is True


def test_the_report_counts_research_only_passes_separately(tmp_path):
    """Lumping them into "rejected" is as wrong as lumping them into "incubating"."""
    from engine.orchestration.population import StrategyRecord
    from engine.reporting.report import _pipeline_section

    population = Population(tmp_path / "population.db")
    rows = [
        ("ensemble:DEXJPUS:1d", "fred", "DEXJPUS", "candidate", True, True),
        ("tsmom:DEXJPUS:1d", "fred", "DEXJPUS", "candidate", False, True),
        ("tsmom:BTC/JPY:1d", "bitbank", "BTC/JPY", "candidate", False, False),
    ]
    for family, source, symbol, status, passed, research_only in rows:
        population.upsert(
            StrategyRecord(
                family=family,
                strategy=family.split(":")[0],
                symbol=symbol,
                source=source,
                timeframe="1d",
                params={},
                status=status,
                metrics={"passed": passed, "research_only": research_only, "ppy": 365.0},
            )
        )
    population.close()

    headline = _pipeline_section(tmp_path)[0]
    assert "rejected: 2" in headline
    assert "research-only passes: 1" in headline
    assert "incubating: 0" in headline
