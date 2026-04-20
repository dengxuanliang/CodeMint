from __future__ import annotations

from pathlib import Path

from codemint.diagnose.item_mode import run_item_mode
from codemint.io.jsonl import read_jsonl
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.custom import build_rules


def test_item_mode_persists_and_returns_expected_diagnoses(tmp_path: Path) -> None:
    tasks = [
        _task(1, "The submission crashes with NameError: helper is not defined"),
        _task(2, "Program returns the wrong total for some inputs."),
    ]
    calls: list[tuple[str, int, str]] = []

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        calls.append(("confirm", task.task_id, rule.rule_id))
        return _diagnosis(
            task.task_id,
            diagnosis_source="rule_confirmed_by_model",
            fault_type=rule.fault_type,
            severity=rule.severity,
            sub_tags=[rule.sub_tag],
            description=f"Confirmed from rule {rule.rule_id}.",
        )

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(("deep", task.task_id, ""))
        return _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            description=f"Deep analysis for task {task.task_id}.",
        )

    output_path = tmp_path / "item.jsonl"
    result = run_item_mode(
        tasks,
        output_path=output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert calls == [("confirm", 1, "R002"), ("deep", 2, "")]
    assert [row.task_id for row in result] == [1, 2]
    assert [row.diagnosis_source for row in result] == ["rule_confirmed_by_model", "model_deep"]
    assert read_jsonl(output_path) == [
        row.model_dump(mode="json") for row in result
    ]


def test_item_mode_preserves_existing_rows_and_skips_completed_tasks(tmp_path: Path) -> None:
    tasks = [
        _task(1, "The submission crashes with NameError: helper is not defined"),
        _task(2, "Program returns the wrong total for some inputs."),
    ]
    existing = _diagnosis(
        1,
        diagnosis_source="rule_only",
        sub_tags=["undefined_variable"],
        severity="medium",
        description="Precomputed diagnosis.",
    )
    output_path = tmp_path / "item.jsonl"
    output_path.write_text(existing.model_dump_json() + "\n", encoding="utf-8")
    calls: list[int] = []

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        calls.append(task.task_id)
        return _diagnosis(task.task_id, diagnosis_source="rule_confirmed_by_model")

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(task.task_id)
        return _diagnosis(task.task_id, diagnosis_source="model_deep")

    result = run_item_mode(
        tasks,
        output_path=output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert [row.task_id for row in result] == [1, 2]
    assert calls == [2]
    assert read_jsonl(output_path)[0] == existing.model_dump(mode="json")
    assert read_jsonl(output_path)[1]["diagnosis_source"] == "model_deep"


def test_item_mode_honors_explicit_empty_rules_list(tmp_path: Path) -> None:
    task = _task(3, "The submission crashes with NameError: helper is not defined")
    output_path = tmp_path / "item.jsonl"
    calls: list[tuple[str, int]] = []

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        calls.append(("confirm", task.task_id))
        return _diagnosis(task.task_id, diagnosis_source="rule_confirmed_by_model")

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(("deep", task.task_id))
        return _diagnosis(task.task_id, diagnosis_source="model_deep")

    result = run_item_mode(
        [task],
        output_path=output_path,
        rules=[],
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert calls == [("deep", 3)]
    assert result[0].diagnosis_source == "model_deep"
    assert read_jsonl(output_path) == [result[0].model_dump(mode="json")]


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
        confidence=0.5,
        diagnosis_source=diagnosis_source,
        prompt_version="test-v1",
    )
