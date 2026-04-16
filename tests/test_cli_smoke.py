import subprocess
import sys
from pathlib import Path
import shutil

import pytest
from typer.testing import CliRunner

from codemint import __version__
from codemint.cli import app


def _python_supports_packaging(executable):
    result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False

    version_text = (result.stdout or result.stderr).strip()
    parts = version_text.removeprefix("Python ").split(".")
    if len(parts) < 2:
        return False

    major, minor = int(parts[0]), int(parts[1])
    return (major, minor) >= (3, 12)


def _find_packaging_python():
    candidates = [sys.executable]
    for name in ("python3.14", "python3.13", "python3.12", "python3"):
        path = shutil.which(name)
        if path and path not in candidates:
            candidates.append(path)

    for candidate in candidates:
        if _python_supports_packaging(candidate):
            return candidate

    return None


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


def test_find_packaging_python_prefers_current_interpreter(monkeypatch):
    def fake_run(args, capture_output, text, check):
        executable = args[0]
        if executable == sys.executable:
            return subprocess.CompletedProcess(args, 0, stdout="Python 3.12.5\n", stderr="")
        raise AssertionError(f"unexpected candidate checked: {executable}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _find_packaging_python() == sys.executable


def test_installed_console_script_shows_help(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    venv_dir = tmp_path / "venv"

    python_executable = _find_packaging_python()
    if python_executable is None:
        pytest.skip("no Python interpreter compatible with requires-python >=3.12")

    create_venv = subprocess.run(
        [python_executable, "-m", "venv", str(venv_dir)],
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
