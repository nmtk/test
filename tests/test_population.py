from engine.orchestration.population import Population, StrategyRecord


def _rec(family="tsmom:BTC/JPY:1h", status="incubating", dsr=0.9, params=None):
    return StrategyRecord(
        family=family,
        strategy="tsmom",
        symbol="BTC/JPY",
        source="bitbank",
        timeframe="1h",
        params=params or {"lookback": 50},
        status=status,
        metrics={"dsr": dsr},
        incubation_start_ts=1000,
    )


def test_insert_get_list(tmp_path):
    pop = Population(tmp_path / "pop.db")
    pop.upsert(_rec())
    assert pop.get("tsmom:BTC/JPY:1h")["status"] == "incubating"
    assert pop.get("missing") is None
    assert len(pop.list()) == 1
    assert len(pop.list("incubating")) == 1
    pop.close()


def test_family_is_unique_key(tmp_path):
    pop = Population(tmp_path / "pop.db")
    pop.upsert(_rec(status="incubating", dsr=0.9, params={"lookback": 50}))
    pop.upsert(_rec(status="champion", dsr=0.97, params={"lookback": 80}))  # same family
    rows = pop.list()
    assert len(rows) == 1  # one row per family
    assert rows[0]["status"] == "champion"
    assert rows[0]["params"] == {"lookback": 80}
    assert rows[0]["metrics"]["dsr"] == 0.97
    pop.close()


def test_update_fields(tmp_path):
    pop = Population(tmp_path / "pop.db")
    pop.upsert(_rec(status="incubating"))
    pop.update_fields("tsmom:BTC/JPY:1h", status="champion", champion_start_ts=5000)
    row = pop.get("tsmom:BTC/JPY:1h")
    assert row["status"] == "champion"
    assert row["champion_start_ts"] == 5000
    assert row["incubation_start_ts"] == 1000  # untouched
    pop.close()
