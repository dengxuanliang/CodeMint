from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from codemint.cli import app
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord


def test_run_aggregate_repairs_clusters_and_writes_report(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(1, "implementation", ["off_by_one", "loop_bound"], confidence=0.9),
        _diagnosis(2, "implementation", ["off_by_one", "indexing"], confidence=0.8),
    ]
    output_path = tmp_path / "weaknesses.json"
    verify_calls: list[tuple[int, str]] = []

    def verify(record: DiagnosisRecord, level: str):
        verify_calls.append((record.task_id, level))
        return {"level": "cross_model", "status": "passed"}

    report = run_aggregate(
        diagnoses,
        output_path,
        verification_level="cross_model",
        verify=verify,
    )

    assert output_path.exists()
    assert verify_calls == [(1, "cross_model"), (2, "cross_model")]
    assert report.weaknesses[0].fault_type == "implementation"
    assert report.weaknesses[0].sub_tags == ["off_by_one"]
    assert report.weaknesses[0].frequency == 2
    assert report.rankings.by_frequency == [1]
    assert report.rankings.by_difficulty == [1]
    assert report.rankings.by_trainability == [1]
    assert report.tag_mappings == {"off_by_one": "off_by_one"}
    written = output_path.read_text(encoding="utf-8")
    assert '"frequency":2' in written


def test_aggregate_command_reads_diagnoses_and_writes_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "demo-run"
    run_dir.mkdir(parents=True)
    diagnoses_path = run_dir / "diagnoses.jsonl"
    diagnoses_path.write_text(
        "\n".join(
            [
                _diagnosis(1, "implementation", ["off_by_one"]).model_dump_json(),
                _diagnosis(2, "implementation", ["off_by_one"]).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "aggregate",
            "--output-root",
            str(tmp_path / "artifacts"),
            "--run-id",
            "demo-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (run_dir / "weaknesses.json").exists()
    assert "Wrote weakness report" in result.stdout


def _diagnosis(
    task_id: int,
    fault_type: str,
    sub_tags: list[str],
    *,
    confidence: float = 0.9,
) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task_id,
        fault_type=fault_type,
        sub_tags=sub_tags,
        severity="medium",
        description=f"Diagnosis {task_id}",
        evidence=DiagnosisEvidence(
            wrong_line="line",
            correct_approach="approach",
            failed_test="test",
        ),
        enriched_labels={},
        confidence=confidence,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )
