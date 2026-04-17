from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from codemint.cli import app
from codemint.io.jsonl import read_jsonl


def test_diagnose_command_loads_tasks_and_writes_diagnoses(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                (
                    '{"task_id": 1, "content": "Task 1", "canonical_solution": "pass", '
                    '"completion": "NameError: helper is not defined", "test_code": "assert True", '
                    '"labels": {}, "accepted": false, "metrics": {}, "extra": {}}'
                ),
                (
                    '{"task_id": 2, "content": "Task 2", "canonical_solution": "pass", '
                    '"completion": "output format mismatch", "test_code": "assert True", '
                    '"labels": {}, "accepted": false, "metrics": {}, "extra": {}}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"

    result = CliRunner().invoke(
        app,
        [
            "diagnose",
            str(input_path),
            "--output-root",
            str(output_root),
            "--run-id",
            "demo-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    diagnoses_path = output_root / "demo-run" / "diagnoses.jsonl"
    assert diagnoses_path.exists()
    assert [row["task_id"] for row in read_jsonl(diagnoses_path)] == [1, 2]
    assert "Wrote 2 new diagnoses" in result.stdout
