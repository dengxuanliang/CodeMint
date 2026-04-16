import subprocess
import sys
from pathlib import Path

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


def test_installed_console_script_shows_help(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    venv_dir = tmp_path / "venv"

    create_venv = subprocess.run(
        ["/opt/homebrew/bin/python3.14", "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert create_venv.returncode == 0, create_venv.stderr

    pip_path = venv_dir / "bin" / "pip"
    install = subprocess.run(
        [str(pip_path), "install", str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr

    cli_path = venv_dir / "bin" / "codemint"
    result = subprocess.run(
        [str(cli_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "diagnose" in result.stdout
    assert "aggregate" in result.stdout
    assert "synthesize" in result.stdout
    assert "run" in result.stdout
