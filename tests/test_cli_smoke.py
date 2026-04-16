import subprocess
import sys

from typer.testing import CliRunner

from codemint import __version__
from codemint.cli import app


def test_cli_shows_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "diagnose" in result.stdout
    assert "aggregate" in result.stdout
    assert "synthesize" in result.stdout
    assert "run" in result.stdout


def test_module_entrypoint_shows_help():
    result = subprocess.run(
        [sys.executable, "-m", "codemint.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "diagnose" in result.stdout
    assert "aggregate" in result.stdout
    assert "synthesize" in result.stdout
    assert "run" in result.stdout


def test_package_exports_version():
    assert __version__ == "0.1.0"
