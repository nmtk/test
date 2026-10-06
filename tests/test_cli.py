from engine.cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert "trade" in capsys.readouterr().out


def test_show_universe(capsys):
    assert main(["show-universe"]) == 0
    out = capsys.readouterr().out
    assert "BTC/JPY" in out
    assert "SPY" in out


def test_list_strategies_includes_archetypes(capsys):
    assert main(["list-strategies"]) == 0
    out = capsys.readouterr().out
    assert "tsmom" in out
    assert "xs_momentum" in out


def test_sync_data_goes_red_when_a_keyless_source_fails(monkeypatch, capsys):
    """A keyless feed failing is a real breakage, not a missing-secret no-op.

    The first FX deployment reported job success while fetching nothing, because every
    per-series exception was swallowed and the exit code only cared about total failure.
    """
    import engine.cli as cli_mod
    from engine.data.sync import SyncResult

    monkeypatch.setattr(
        cli_mod,
        "sync_universe",
        lambda *a, **k: [
            SyncResult("bitbank", "BTC/JPY", "1h", 100, ok=True),
            SyncResult("dukascopy", "USD/JPY", "5m", 0, ok=False, error="connect timeout"),
        ],
    )
    monkeypatch.setattr(cli_mod, "sync_context", lambda *a, **k: [])
    monkeypatch.setattr(cli_mod, "sync_intrabar", lambda *a, **k: [])

    code = cli_mod.main(["sync-data"])
    out = capsys.readouterr().out
    assert code == 1, "a keyless failure must fail the job"
    assert "dukascopy:USD/JPY:5m" in out


def test_sync_data_goes_red_when_the_intrabar_path_fails(monkeypatch, capsys):
    """The path comes from the same keyless feed, so its failure is just as real."""
    import engine.cli as cli_mod
    from engine.data.intrabar import PATH_LABEL
    from engine.data.sync import SyncResult

    monkeypatch.setattr(
        cli_mod,
        "sync_universe",
        lambda *a, **k: [SyncResult("dukascopy", "USD/JPY", "5m", 100, ok=True)],
    )
    monkeypatch.setattr(cli_mod, "sync_context", lambda *a, **k: [])
    monkeypatch.setattr(
        cli_mod,
        "sync_intrabar",
        lambda *a, **k: [
            SyncResult("dukascopy", "USD/JPY", PATH_LABEL, 0, ok=False, error="timeout")
        ],
    )

    assert cli_mod.main(["sync-data"]) == 1
    assert f"dukascopy:USD/JPY:{PATH_LABEL}" in capsys.readouterr().out


def test_sync_data_tolerates_an_unconfigured_credentialed_venue(monkeypatch, capsys):
    """Running without Alpaca keys is the normal state, not a failure."""
    import engine.cli as cli_mod
    from engine.data.sync import SyncResult

    monkeypatch.setattr(
        cli_mod,
        "sync_universe",
        lambda *a, **k: [
            SyncResult("bitbank", "BTC/JPY", "1h", 100, ok=True),
            SyncResult("alpaca", "SPY", "1h", 0, ok=False, error="must supply authentication"),
        ],
    )
    monkeypatch.setattr(cli_mod, "sync_context", lambda *a, **k: [])
    monkeypatch.setattr(cli_mod, "sync_intrabar", lambda *a, **k: [])
    assert cli_mod.main(["sync-data"]) == 0


def test_stooq_stays_best_effort(monkeypatch):
    """stooq 404s runners by design here, so it must not hold the job red."""
    from engine.data.sources import KEYLESS_SOURCES

    assert "stooq" not in KEYLESS_SOURCES
    assert {"fred", "dukascopy"} <= KEYLESS_SOURCES
