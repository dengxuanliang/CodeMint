from typer.testing import CliRunner

from codemint.cli import app


def test_cli_shows_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "diagnose" in result.stdout
    assert "aggregate" in result.stdout
    assert "synthesize" in result.stdout
    assert "run" in result.stdout
