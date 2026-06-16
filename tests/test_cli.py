from engine.cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert "trade" in capsys.readouterr().out


def test_show_universe(capsys):
    assert main(["show-universe"]) == 0
    out = capsys.readouterr().out
    assert "BTC/JPY" in out
    assert "SPY" in out


def test_stub_commands_exit_zero(capsys):
    for cmd in ("paper-step", "research", "report"):
        assert main([cmd]) == 0
        assert "not implemented" in capsys.readouterr().out


def test_list_strategies_includes_archetypes(capsys):
    assert main(["list-strategies"]) == 0
    out = capsys.readouterr().out
    assert "tsmom" in out
    assert "xs_momentum" in out
