from __future__ import annotations

from pathlib import Path

from codemint.diagnose.pipeline import run_diagnose
from codemint.io.jsonl import read_jsonl
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.custom import build_rules


def test_rule_match_with_high_severity_calls_confirmation_model(tmp_path: Path) -> None:
    task = _task(
        1,
        "The submission crashes with NameError: helper is not defined",
    )
    output_path = tmp_path / "diagnoses.jsonl"
    calls: list[tuple[str, int, str]] = []

    def confirm_analyzer(task: TaskRecord, rule_id: str) -> DiagnosisRecord:
        calls.append(("confirm", task.task_id, rule_id))
        return _diagnosis(
            task.task_id,
            diagnosis_source="rule_confirmed_by_model",
            fault_type="surface",
            severity="medium",
            description="Confirmed undefined variable issue.",
            sub_tags=["undefined_variable", "confirmed"],
        )

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(("deep", task.task_id, ""))
        return _diagnosis(task.task_id, diagnosis_source="model_deep")

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert [diagnosis.task_id for diagnosis in diagnoses] == [1]
    assert calls == [("confirm", 1, "R002")]
    assert diagnoses[0].diagnosis_source == "rule_confirmed_by_model"
    assert read_jsonl(output_path) == [diagnoses[0].model_dump(mode="json")]


def test_assertion_error_routes_to_deep_analysis(tmp_path: Path) -> None:
    task = _task(7, "Tests fail with AssertionError: expected 3 == 4")
    output_path = tmp_path / "diagnoses.jsonl"
    calls: list[tuple[str, int]] = []

    def confirm_analyzer(task: TaskRecord, rule_id: str) -> DiagnosisRecord:
        calls.append((f"confirm:{rule_id}", task.task_id))
        return _diagnosis(task.task_id, diagnosis_source="rule_confirmed_by_model")

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(("deep", task.task_id))
        return _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            description="Deep analysis classified the assertion failure.",
        )

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert [diagnosis.task_id for diagnosis in diagnoses] == [7]
    assert calls == [("deep", 7)]
    assert diagnoses[0].diagnosis_source == "model_deep"


def test_existing_diagnosis_rows_are_preserved_and_only_missing_tasks_are_processed(
    tmp_path: Path,
) -> None:
    tasks = [
        _task(1, "NameError: helper is not defined"),
        _task(2, "output format mismatch in the final line"),
    ]
    output_path = tmp_path / "diagnoses.jsonl"
    existing = _diagnosis(
        1,
        diagnosis_source="rule_only",
        description="Precomputed diagnosis.",
    )
    output_path.write_text(existing.model_dump_json() + "\n", encoding="utf-8")
    calls: list[int] = []

    def confirm_analyzer(task: TaskRecord, rule_id: str) -> DiagnosisRecord:
        calls.append(task.task_id)
        return _diagnosis(
            task.task_id,
            diagnosis_source="rule_confirmed_by_model",
            severity="medium",
            description=f"Confirmed from rule {rule_id}.",
        )

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(task.task_id)
        return _diagnosis(task.task_id, diagnosis_source="model_deep")

    diagnoses = run_diagnose(
        tasks,
        output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert [diagnosis.task_id for diagnosis in diagnoses] == [1, 2]
    assert calls == []
    assert [row["task_id"] for row in read_jsonl(output_path)] == [1, 2]
    assert read_jsonl(output_path)[0] == existing.model_dump(mode="json")
    assert read_jsonl(output_path)[1]["diagnosis_source"] == "rule_only"


def _task(task_id: int, completion: str) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        content=f"Task {task_id}",
        canonical_solution="pass",
        completion=completion,
        test_code="assert True",
        labels={},
        accepted=False,
        metrics={},
        extra={},
    )


def _diagnosis(
    task_id: int,
    *,
    diagnosis_source: str,
    fault_type: str = "implementation",
    sub_tags: list[str] | None = None,
    severity: str = "low",
    description: str = "Diagnosis",
) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task_id,
        fault_type=fault_type,
        sub_tags=sub_tags or ["stub"],
        severity=severity,
        description=description,
        evidence=DiagnosisEvidence(
            wrong_line="line",
            correct_approach="approach",
            failed_test="test",
        ),
        enriched_labels={},
        confidence=0.9,
        diagnosis_source=diagnosis_source,
        prompt_version="test-v1",
    )
