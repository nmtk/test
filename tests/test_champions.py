from engine.orchestration.champions import Allocation, Champions, load_champions, save_champions


def test_missing_file_is_empty(tmp_path):
    champs = load_champions(tmp_path / "champions.json")
    assert champs.allocations == []


def test_roundtrip(tmp_path):
    champs = Champions(
        allocations=[
            Allocation("tsmom", "BTC/JPY", "bitbank", "1h", 0.5, {"lookback": 100}),
            Allocation("donchian", "ETH/JPY", "bitbank", "4h", 0.5, {"window": 20}),
        ],
        updated="2026-06-16T00:00:00+00:00",
    )
    path = tmp_path / "champions.json"
    save_champions(path, champs)
    loaded = load_champions(path)
    assert len(loaded.allocations) == 2
    a = loaded.allocations[0]
    assert a.strategy == "tsmom"
    assert a.symbol == "BTC/JPY"
    assert a.weight == 0.5
    assert a.params == {"lookback": 100}
    assert loaded.updated == "2026-06-16T00:00:00+00:00"


# -- the no-op write ------------------------------------------------------------------
#
# Three workflows run `trade promote` and commit state/: executor-paper (hourly),
# executor-live, and research-nightly. save_champions used to rewrite the file on every
# call, so each run stamped a new `updated` on the same line. An hourly paper step landing
# inside the nightly research window then produced a guaranteed rebase conflict on
# champions.json, and the nightly's push guard discarded the entire twenty-minute research
# run rather than guess. It happened on 2026-10-31 and again on 2026-11-06, both times with
# identical (empty) allocations on both sides.


def _alloc(weight: float = 0.5) -> Allocation:
    return Allocation("tsmom", "BTC/JPY", "bitbank", "1h", weight, {"lookback": 100})


def test_an_unchanged_save_does_not_touch_the_file(tmp_path):
    path = tmp_path / "champions.json"
    save_champions(path, Champions(allocations=[_alloc()], updated="2026-06-16T00:00:00+00:00"))
    before = path.read_bytes()
    mtime = path.stat().st_mtime_ns

    # promote runs again an hour later and computes exactly the same allocations
    save_champions(path, Champions(allocations=[_alloc()], updated="2026-06-16T01:00:00+00:00"))
    assert path.read_bytes() == before, "a fresh timestamp alone must not rewrite the file"
    assert path.stat().st_mtime_ns == mtime, "the file must not be touched at all"


def test_empty_allocations_also_stop_churning(tmp_path):
    """The exact observed case: both sides empty, and still conflicting every hour."""
    path = tmp_path / "champions.json"
    save_champions(path, Champions(allocations=[], updated="2026-11-06T22:59:00+00:00"))
    before = path.read_bytes()
    save_champions(path, Champions(allocations=[], updated="2026-11-06T23:10:00+00:00"))
    assert path.read_bytes() == before


def test_a_real_change_is_written(tmp_path):
    path = tmp_path / "champions.json"
    save_champions(path, Champions(allocations=[_alloc(0.5)], updated="2026-06-16T00:00:00+00:00"))
    save_champions(path, Champions(allocations=[_alloc(0.25)], updated="2026-06-17T00:00:00+00:00"))
    loaded = load_champions(path)
    assert loaded.allocations[0].weight == 0.25
    assert loaded.updated == "2026-06-17T00:00:00+00:00", "updated tracks the last real change"


def test_params_are_part_of_the_comparison(tmp_path):
    """Same strategy, symbol and weight, different params: that is a different champion."""
    path = tmp_path / "champions.json"
    save_champions(
        path,
        Champions(
            allocations=[Allocation("tsmom", "BTC/JPY", "bitbank", "1h", 0.5, {"lookback": 100})],
            updated="a",
        ),
    )
    save_champions(
        path,
        Champions(
            allocations=[Allocation("tsmom", "BTC/JPY", "bitbank", "1h", 0.5, {"lookback": 20})],
            updated="b",
        ),
    )
    assert load_champions(path).allocations[0].params == {"lookback": 20}


def test_added_and_removed_allocations_are_written(tmp_path):
    path = tmp_path / "champions.json"
    save_champions(path, Champions(allocations=[_alloc()], updated="a"))
    save_champions(path, Champions(allocations=[_alloc(), _alloc(0.1)], updated="b"))
    assert len(load_champions(path).allocations) == 2
    save_champions(path, Champions(allocations=[], updated="c"))
    assert load_champions(path).allocations == []


def test_a_corrupt_file_is_rewritten_rather_than_trusted(tmp_path):
    path = tmp_path / "champions.json"
    path.write_text("{not json")
    save_champions(path, Champions(allocations=[_alloc()], updated="2026-06-16T00:00:00+00:00"))
    assert load_champions(path).allocations[0].weight == 0.5
