from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from codemint.cli import app
from codemint.io.jsonl import append_jsonl, read_jsonl
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.spec import (
    DiversityTags,
    GenerationHints,
    LanguageConstraint,
    ProblemConstraints,
    ProblemSpec,
    SpecRecord,
    TargetWeakness,
    VerificationSpec,
)
from codemint.models.task import TaskRecord
from codemint.models.weakness import CausalChain, CollectiveDiagnosis, RankingSet, WeaknessEntry, WeaknessReport
from codemint.run.pipeline import run_pipeline


def test_run_skips_complete_stage_and_gap_fills_incomplete_stage(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                _task(1, "NameError: helper is not defined"),
                _task(2, "Wrong answer on hidden case"),
                _task(3, "Output format mismatch"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    run_dir = output_root / "demo-run"
    run_dir.mkdir(parents=True)
    append_jsonl(
        run_dir / "diagnoses.jsonl",
        [
            _diagnosis(1).model_dump(mode="json"),
            _diagnosis(3, source="model_deep").model_dump(mode="json"),
        ],
    )
    (run_dir / "weaknesses.json").write_text(_report().model_dump_json(), encoding="utf-8")
    append_jsonl(run_dir / "specs.jsonl", [_spec().model_dump(mode="json")])

    diagnose_calls: list[list[int]] = []
    aggregate_calls: list[list[int]] = []
    synthesize_calls: list[list[int]] = []

    result = run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="demo-run",
        run_diagnose_stage=lambda tasks, output_path: _record_diagnose(diagnose_calls, tasks, output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _record_aggregate(aggregate_calls, diagnoses, output_path),
        run_synthesize_stage=lambda report, output_path: _record_synthesize(synthesize_calls, report, output_path),
    )

    assert result.stages_executed == ["diagnose", "aggregate", "synthesize"]
    assert diagnose_calls == [[1, 2, 3]]
    assert aggregate_calls == [[1, 3, 2]]
    assert synthesize_calls == [[1]]
    assert [row["task_id"] for row in read_jsonl(run_dir / "diagnoses.jsonl")] == [1, 3, 2]
    assert json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))["summary"] == {
        "diagnosed": 3,
        "rule_screened": 1,
        "model_analyzed": 2,
        "errors": 0,
        "weaknesses_found": 1,
        "specs_generated": 1,
    }


def test_run_start_from_aggregate_skips_diagnose_and_forces_downstream_rerun(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                _task(1, "NameError: helper is not defined"),
                _task(2, "Wrong answer on hidden case"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    run_dir = output_root / "demo-run"
    run_dir.mkdir(parents=True)
    append_jsonl(
        run_dir / "diagnoses.jsonl",
        [_diagnosis(1).model_dump(mode="json"), _diagnosis(2, source="model_deep").model_dump(mode="json")],
    )
    (run_dir / "weaknesses.json").write_text(_report().model_dump_json(), encoding="utf-8")
    append_jsonl(run_dir / "specs.jsonl", [_spec().model_dump(mode="json")])

    diagnose_calls: list[list[int]] = []
    aggregate_calls: list[list[int]] = []
    synthesize_calls: list[list[int]] = []

    result = run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="demo-run",
        start_from="aggregate",
        run_diagnose_stage=lambda tasks, output_path: _record_diagnose(diagnose_calls, tasks, output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _record_aggregate(aggregate_calls, diagnoses, output_path),
        run_synthesize_stage=lambda report, output_path: _record_synthesize(synthesize_calls, report, output_path),
    )

    assert result.stages_executed == ["aggregate", "synthesize"]
    assert diagnose_calls == []
    assert aggregate_calls == [[1, 2]]
    assert synthesize_calls == [[1]]


def test_run_cli_from_override_wires_stage_control_and_rich_dry_run(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                _task(1, "NameError: helper is not defined"),
                _task(2, "Wrong answer on hidden case"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    run_dir = output_root / "demo-run"
    run_dir.mkdir(parents=True)
    append_jsonl(
        run_dir / "diagnoses.jsonl",
        [_diagnosis(1).model_dump(mode="json"), _diagnosis(2, source="model_deep").model_dump(mode="json")],
    )
    (run_dir / "weaknesses.json").write_text(_report().model_dump_json(), encoding="utf-8")
    append_jsonl(run_dir / "specs.jsonl", [_spec().model_dump(mode="json")])
    runner = CliRunner()

    dry_run_result = runner.invoke(app, ["run", str(input_path), "--dry-run"])

    assert dry_run_result.exit_code == 0, dry_run_result.stdout
    assert "Dry run total:" in dry_run_result.stdout
    assert "diagnose: 2 calls" in dry_run_result.stdout
    assert "aggregate: 1 call" in dry_run_result.stdout
    assert "synthesize: 2 calls" in dry_run_result.stdout

    from_result = runner.invoke(
        app,
        [
            "run",
            str(input_path),
            "--output-root",
            str(output_root),
            "--run-id",
            "demo-run",
            "--from",
            "aggregate",
        ],
    )

    assert from_result.exit_code == 0, from_result.stdout
    assert "stages=aggregate,synthesize" in from_result.stdout


def _record_diagnose(
    calls: list[list[int]],
    tasks: list[TaskRecord],
    output_path: Path,
) -> list[DiagnosisRecord]:
    calls.append([task.task_id for task in tasks])
    append_jsonl(output_path, [_diagnosis(2, source="model_deep").model_dump(mode="json")])
    return [_diagnosis(1), _diagnosis(3, source="model_deep"), _diagnosis(2, source="model_deep")]


def _record_aggregate(
    calls: list[list[int]],
    diagnoses: list[DiagnosisRecord],
    output_path: Path,
) -> WeaknessReport:
    calls.append([diagnosis.task_id for diagnosis in diagnoses])
    report = _report()
    output_path.write_text(report.model_dump_json(), encoding="utf-8")
    return report


def _record_synthesize(
    calls: list[list[int]],
    report: WeaknessReport,
    output_path: Path,
) -> list[SpecRecord]:
    calls.append([entry.rank for entry in report.weaknesses])
    append_jsonl(output_path, [_spec().model_dump(mode="json")])
    return [_spec()]


def _task(task_id: int, completion: str) -> str:
    return json.dumps(
        {
            "task_id": task_id,
            "content": f"Task {task_id}",
            "canonical_solution": "pass",
            "completion": completion,
            "test_code": "assert True",
            "labels": {},
            "accepted": False,
            "metrics": {},
            "extra": {},
        }
    )


def _diagnosis(task_id: int, *, source: str = "rule_only") -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task_id,
        fault_type="implementation",
        sub_tags=["stub"],
        severity="low",
        description=f"Diagnosis {task_id}",
        evidence=DiagnosisEvidence(
            wrong_line="line",
            correct_approach="approach",
            failed_test="test",
        ),
        enriched_labels={},
        confidence=0.9,
        diagnosis_source=source,
        prompt_version="diagnose-test-v1",
    )


def _report() -> WeaknessReport:
    return WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="implementation",
                sub_tags=["stub"],
                frequency=3,
                sample_task_ids=[1, 2, 3],
                trainability=0.6,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Root cause",
                    capability_cliff="Capability cliff",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=1.0,
                ),
            )
        ],
        rankings=RankingSet(by_frequency=[1], by_difficulty=[1], by_trainability=[1]),
        causal_chains=[CausalChain(root="stub", downstream=[], training_priority="high")],
        tag_mappings={"stub": "stub"},
    )


def _spec() -> SpecRecord:
    return SpecRecord(
        spec_id="spec-0001",
        target_weakness=TargetWeakness(
            fault_type="implementation",
            sub_tags=["stub"],
            root_cause="Root cause",
            capability_cliff="Capability cliff",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="array",
            difficulty="medium",
            narrative_theme="factories",
            constraints=ProblemConstraints(
                n_range=[1, 10],
                value_range=[0, 100],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="trap",
            must_cover=["stub"],
            must_avoid=["duplicate"],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=3,
            must_include_edge_cases=["n=1"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="factories",
            data_structure="array",
            constraint_scale="small",
        ),
        generation_hints=GenerationHints(
            solution_approach="Approach",
            common_wrong_approach="Wrong",
            distinguishing_test="Test",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="synthesize-test-v1",
    )
