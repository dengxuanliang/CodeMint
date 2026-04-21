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
from codemint.config import CodeMintConfig


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
    summary = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))["summary"]
    assert summary == {
        "diagnosed": 3,
        "rule_screened": 1,
        "model_analyzed": 2,
        "non_failures": 0,
        "errors": 0,
        "skipped": 0,
        "elapsed_seconds": summary["elapsed_seconds"],
        "weaknesses_found": 1,
        "specs_generated": 1,
        "synthesize_failures": 0,
        "specs_by_weakness": {"stub": 1},
        "synthesize_status": "success",
        "attempted_weaknesses": ["stub"],
        "covered_weaknesses": ["stub"],
        "weaknesses_without_specs": [],
        "synthesize_failure_reasons_by_weakness": {},
        "diagnose_processing_mode": "item",
        "cluster_count": 0,
        "compression_ratio": 1.0,
        "representative_diagnoses": 0,
        "propagated_diagnoses": 0,
        "fallback_item_diagnoses": 0,
    }
    assert summary["elapsed_seconds"] >= 0


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


def test_run_pipeline_emits_progress_events_for_started_skipped_and_completed_stages(tmp_path: Path) -> None:
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
    events: list[dict[str, object]] = []

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="demo-run",
        progress_callback=events.append,
        run_aggregate_stage=lambda diagnoses, output_path: _record_aggregate([], diagnoses, output_path),
        run_synthesize_stage=lambda report, output_path: _record_synthesize([], report, output_path),
    )

    assert [event["stage"] for event in events] == ["diagnose", "aggregate", "aggregate", "synthesize", "synthesize"]
    assert events[0]["status"] == "skipped"
    assert events[1]["status"] == "started"
    assert events[2]["status"] == "completed"
    assert events[2]["processed"] == 2
    assert events[4]["status"] == "completed"
    assert events[4]["total"] == 1


def test_run_pipeline_uses_spec_slot_total_for_synthesize_progress(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        _task(1, "Wrong answer on hidden case") + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    events: list[dict[str, object]] = []

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="slot-total-run",
        progress_callback=events.append,
        run_diagnose_stage=lambda tasks, output_path: _record_diagnose([], tasks, output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _record_aggregate([], diagnoses, output_path),
        run_synthesize_stage=lambda report, output_path: [_make_spec("spec-0001"), _make_spec("spec-0002")],
    )

    synthesize_events = [event for event in events if event["stage"] == "synthesize"]
    assert synthesize_events[0]["total"] == 1
    assert synthesize_events[-1]["status"] == "completed"
    assert synthesize_events[-1]["processed"] == 2
    assert synthesize_events[-1]["total"] == 2


def test_run_pipeline_passes_config_to_default_diagnose_stage(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(_task(1, "Wrong answer on hidden case") + "\n", encoding="utf-8")
    output_root = tmp_path / "artifacts"
    seen: dict[str, object] = {"called": False, "config": None}

    def fake_run_diagnose(tasks, output_path, **kwargs):
        seen["called"] = True
        seen["config"] = kwargs.get("config")
        append_jsonl(output_path, [_diagnosis(1, source="model_deep").model_dump(mode="json")])
        return [_diagnosis(1, source="model_deep")]

    monkeypatch.setattr("codemint.run.pipeline.run_diagnose", fake_run_diagnose)

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="config-pass-run",
        config=CodeMintConfig.model_validate({"model": {"analysis_model": "gpt-test"}}),
        run_aggregate_stage=lambda diagnoses, output_path: _record_aggregate([], diagnoses, output_path),
        run_synthesize_stage=lambda report, output_path: _record_synthesize([], report, output_path),
    )

    assert seen["called"] is True
    assert isinstance(seen["config"], CodeMintConfig)


def test_run_pipeline_includes_clustered_diagnose_summary_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                _task(1, "def solve(x):\n    return x + 1"),
                _task(2, "def solve(x):\n    return x + 1"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="clustered-run",
        config=CodeMintConfig.model_validate(
            {
                "diagnose": {
                    "processing_mode": "clustered",
                    "cluster_representatives": 1,
                    "rediagnose_low_confidence": False,
                }
            }
        ),
        run_aggregate_stage=lambda diagnoses, output_path: _record_aggregate([], diagnoses, output_path),
        run_synthesize_stage=lambda report, output_path: _record_synthesize([], report, output_path),
    )

    summary = json.loads((output_root / "clustered-run" / "run_metadata.json").read_text(encoding="utf-8"))[
        "summary"
    ]
    assert summary["diagnose_processing_mode"] == "clustered"
    assert summary["cluster_count"] == 1
    assert summary["compression_ratio"] == 2.0
    assert summary["representative_diagnoses"] == 1
    assert summary["propagated_diagnoses"] == 1
    assert summary["fallback_item_diagnoses"] == 0


def test_run_cli_formats_rich_progress_lines(tmp_path: Path) -> None:
    from codemint import cli as cli_module

    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(_task(1, "Wrong answer on hidden case") + "\n", encoding="utf-8")
    output_root = tmp_path / "artifacts"
    runner = CliRunner()
    original_run_pipeline = cli_module.run_pipeline

    def fake_run_pipeline(*, input_paths, output_root, run_id, start_from, config, progress_callback):
        progress_callback = fake_run_pipeline.progress_callback
        progress_callback(
            {"stage": "diagnose", "status": "started", "processed": 0, "total": 3, "errors": 0, "eta_seconds": 9}
        )
        progress_callback(
            {"stage": "diagnose", "status": "completed", "processed": 3, "total": 3, "errors": 1, "eta_seconds": 0}
        )
        return run_pipeline(
            input_paths=input_paths,
            output_root=output_root,
            run_id=run_id,
            start_from=start_from,
            config=config,
            run_diagnose_stage=lambda tasks, output_path: _record_diagnose([], tasks, output_path),
            run_aggregate_stage=lambda diagnoses, output_path: _record_aggregate([], diagnoses, output_path),
            run_synthesize_stage=lambda report, output_path: _record_synthesize([], report, output_path),
        )

    fake_run_pipeline.progress_callback = lambda event: None

    def capture_run_pipeline(**kwargs):
        fake_run_pipeline.progress_callback = kwargs["progress_callback"]
        return fake_run_pipeline(**kwargs)

    cli_module.run_pipeline = capture_run_pipeline
    try:
        result = runner.invoke(
            app,
            [
                "run",
                str(input_path),
                "--output-root",
                str(output_root),
                "--run-id",
                "rich-progress-run",
            ],
        )
    finally:
        cli_module.run_pipeline = original_run_pipeline

    assert result.exit_code == 0, result.stdout
    assert "[diagnose]" in result.stdout
    assert "0/3 (0%)" in result.stdout
    assert "3/3 (100%)" in result.stdout
    assert "1 errors" in result.stdout
    assert "ETA 9s" in result.stdout


def test_run_cli_suppresses_duplicate_progress_lines(tmp_path: Path) -> None:
    from codemint import cli as cli_module

    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(_task(1, "Wrong answer on hidden case") + "\n", encoding="utf-8")
    output_root = tmp_path / "artifacts"
    runner = CliRunner()
    original_run_pipeline = cli_module.run_pipeline

    def fake_run_pipeline(*, input_paths, output_root, run_id, start_from, config, progress_callback):
        duplicate = {
            "stage": "diagnose",
            "status": "running",
            "processed": 1,
            "total": 2,
            "errors": 0,
            "eta_seconds": 3,
        }
        progress_callback(duplicate)
        progress_callback(dict(duplicate))
        progress_callback(
            {
                "stage": "diagnose",
                "status": "completed",
                "processed": 2,
                "total": 2,
                "errors": 0,
                "eta_seconds": 0,
            }
        )
        return run_pipeline(
            input_paths=input_paths,
            output_root=output_root,
            run_id=run_id,
            start_from=start_from,
            config=config,
            run_diagnose_stage=lambda tasks, output_path: _record_diagnose([], tasks, output_path),
            run_aggregate_stage=lambda diagnoses, output_path: _record_aggregate([], diagnoses, output_path),
            run_synthesize_stage=lambda report, output_path: _record_synthesize([], report, output_path),
        )

    cli_module.run_pipeline = fake_run_pipeline
    try:
        result = runner.invoke(
            app,
            [
                "run",
                str(input_path),
                "--output-root",
                str(output_root),
                "--run-id",
                "dedupe-progress-run",
            ],
        )
    finally:
        cli_module.run_pipeline = original_run_pipeline

    assert result.exit_code == 0, result.stdout
    assert result.stdout.count("[diagnose]") == 2


def test_run_cli_summary_shows_degraded_synthesize_status() -> None:
    from datetime import UTC, datetime

    from codemint.logging import format_run_summary
    from codemint.models.run_metadata import PromptVersions, RunMetadata, RunSummary

    metadata = RunMetadata(
        run_id="demo",
        timestamp=datetime.now(UTC),
        config_snapshot={},
        analysis_model="gpt-test",
        prompt_versions=PromptVersions(diagnose="v1", aggregate="v1", synthesize="v1"),
        input_files=["tasks.jsonl"],
        input_count=2,
        stages_executed=["diagnose", "aggregate", "synthesize"],
        summary=RunSummary(
            diagnosed=2,
            rule_screened=1,
            model_analyzed=1,
            non_failures=0,
            errors=2,
            skipped=0,
            elapsed_seconds=1.2,
            weaknesses_found=2,
            specs_generated=1,
            synthesize_failures=1,
            specs_by_weakness={"loop_bound": 1},
            synthesize_status="degraded",
            attempted_weaknesses=["loop_bound", "missing_code"],
            covered_weaknesses=["loop_bound"],
            weaknesses_without_specs=["missing_code"],
            synthesize_failure_reasons_by_weakness={"missing_code": ["boom"]},
        ),
    )

    summary = format_run_summary(metadata)

    assert "synth_status=degraded" in summary
    assert "missing=missing_code" in summary


def test_run_reruns_downstream_when_existing_outputs_are_incomplete(tmp_path: Path) -> None:
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
    incomplete_report = WeaknessReport(
        weaknesses=[],
        rankings=RankingSet(by_frequency=[], by_difficulty=[], by_trainability=[]),
        causal_chains=[],
        tag_mappings={},
    )
    (run_dir / "weaknesses.json").write_text(incomplete_report.model_dump_json(), encoding="utf-8")
    append_jsonl(run_dir / "specs.jsonl", [])

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


def test_run_reruns_incomplete_downstream_artifacts_even_when_diagnose_is_complete(tmp_path: Path) -> None:
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
        [
            _diagnosis(1).model_dump(mode="json"),
            _diagnosis(2, source="model_deep").model_dump(mode="json"),
        ],
    )
    incomplete_report = WeaknessReport(
        weaknesses=[],
        rankings=RankingSet(by_frequency=[], by_difficulty=[], by_trainability=[]),
        causal_chains=[],
        tag_mappings={},
    )
    (run_dir / "weaknesses.json").write_text(incomplete_report.model_dump_json(), encoding="utf-8")
    append_jsonl(run_dir / "specs.jsonl", [])

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
    assert "Estimated model calls:" in dry_run_result.stdout
    assert "Rule-screened (no model call):" in dry_run_result.stdout
    assert "Estimated time:" in dry_run_result.stdout
    assert "diagnose: 1 call" in dry_run_result.stdout
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


def test_run_cli_emits_stage_progress_lines(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(_task(1, "Wrong answer on hidden case") + "\n", encoding="utf-8")
    output_root = tmp_path / "artifacts"
    runner = CliRunner()

    from codemint import cli as cli_module

    original_run_pipeline = cli_module.run_pipeline

    def fake_run_pipeline(*, input_paths, output_root, run_id, start_from, config, progress_callback):
        return run_pipeline(
            input_paths=input_paths,
            output_root=output_root,
            run_id=run_id,
            start_from=start_from,
            config=config,
            progress_callback=progress_callback,
            run_diagnose_stage=lambda tasks, output_path: _record_diagnose([], tasks, output_path),
            run_aggregate_stage=lambda diagnoses, output_path: _record_aggregate([], diagnoses, output_path),
            run_synthesize_stage=lambda report, output_path: _record_synthesize([], report, output_path),
        )

    cli_module.run_pipeline = fake_run_pipeline
    try:
        result = runner.invoke(
            app,
            [
                "run",
                str(input_path),
                "--output-root",
                str(output_root),
                "--run-id",
                "progress-run",
            ],
        )
    finally:
        cli_module.run_pipeline = original_run_pipeline

    assert result.exit_code == 0, result.stdout
    assert "[diagnose]" in result.stdout
    assert "[aggregate]" in result.stdout
    assert "[synthesize]" in result.stdout


def test_run_cli_emits_multiple_progress_updates_with_real_pipeline_events(tmp_path: Path) -> None:
    from codemint import cli as cli_module

    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                _task(1, "Wrong answer on hidden case"),
                _task(2, "Program returns the wrong total for some inputs."),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    runner = CliRunner()
    original_run_pipeline = cli_module.run_pipeline

    def fake_run_pipeline(*, input_paths, output_root, run_id, start_from, config, progress_callback):
        progress_callback(
            {"stage": "diagnose", "status": "started", "processed": 0, "total": 2, "errors": 0, "eta_seconds": 6}
        )
        progress_callback(
            {"stage": "diagnose", "status": "running", "processed": 1, "total": 2, "errors": 0, "eta_seconds": 3}
        )
        progress_callback(
            {"stage": "diagnose", "status": "completed", "processed": 2, "total": 2, "errors": 0, "eta_seconds": 0}
        )
        return run_pipeline(
            input_paths=input_paths,
            output_root=output_root,
            run_id=run_id,
            start_from=start_from,
            config=config,
            progress_callback=progress_callback,
            run_diagnose_stage=lambda tasks, output_path: _record_diagnose([], tasks, output_path),
            run_aggregate_stage=lambda diagnoses, output_path: _record_aggregate([], diagnoses, output_path),
            run_synthesize_stage=lambda report, output_path: _record_synthesize([], report, output_path),
        )

    cli_module.run_pipeline = fake_run_pipeline

    try:
        result = runner.invoke(
            app,
            [
                "run",
                str(input_path),
                "--output-root",
                str(output_root),
                "--run-id",
                "fine-progress-run",
            ],
        )
    finally:
        cli_module.run_pipeline = original_run_pipeline

    assert result.exit_code == 0, result.stdout
    assert result.stdout.count("[diagnose]") >= 3


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
    return _make_spec("spec-0001")


def _make_spec(spec_id: str) -> SpecRecord:
    return SpecRecord(
        spec_id=spec_id,
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
