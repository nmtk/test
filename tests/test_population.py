from engine.orchestration.population import Population, StrategyRecord


def _rec(params, status="challenger", dsr=0.9):
    return StrategyRecord(
        family="tsmom:BTC/JPY:1h",
        strategy="tsmom",
        symbol="BTC/JPY",
        source="bitbank",
        timeframe="1h",
        params=params,
        status=status,
        metrics={"dsr": dsr},
    )


def test_insert_and_list(tmp_path):
    pop = Population(tmp_path / "pop.db")
    pop.upsert(_rec({"lookback": 50}))
    pop.upsert(_rec({"lookback": 100}, status="candidate"))
    assert len(pop.list()) == 2
    assert len(pop.list("challenger")) == 1
    assert len(pop.list("candidate")) == 1
    pop.close()


def test_upsert_updates_same_family_params(tmp_path):
    pop = Population(tmp_path / "pop.db")
    pop.upsert(_rec({"lookback": 50}, status="challenger", dsr=0.9))
    pop.upsert(_rec({"lookback": 50}, status="champion", dsr=0.97))  # same key
    rows = pop.list()
    assert len(rows) == 1  # updated, not duplicated
    assert rows[0]["status"] == "champion"
    assert rows[0]["metrics"]["dsr"] == 0.97
    pop.close()


def test_set_status(tmp_path):
    pop = Population(tmp_path / "pop.db")
    pop.upsert(_rec({"lookback": 50}))
    pop.set_status("tsmom:BTC/JPY:1h", {"lookback": 50}, "retired")
    assert len(pop.list("retired")) == 1
    assert len(pop.list("challenger")) == 0
    pop.close()
