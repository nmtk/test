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
