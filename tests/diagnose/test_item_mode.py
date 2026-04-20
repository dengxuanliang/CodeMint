from __future__ import annotations

from pathlib import Path

from codemint.diagnose.item_mode import run_item_mode
from codemint.diagnose.pipeline import run_diagnose
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.custom import build_rules


def test_item_mode_matches_existing_pipeline_behavior(tmp_path: Path) -> None:
    tasks = [
        _task(1, "The submission crashes with NameError: helper is not defined"),
        _task(2, "Program returns the wrong total for some inputs."),
    ]

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        return _diagnosis(
            task.task_id,
            diagnosis_source="rule_confirmed_by_model",
            fault_type=rule.fault_type,
            severity=rule.severity,
            sub_tags=[rule.sub_tag],
            description=f"Confirmed from rule {rule.rule_id}.",
        )

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        return _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            description=f"Deep analysis for task {task.task_id}.",
        )

    pipeline_result = run_diagnose(
        tasks,
        tmp_path / "diagnoses.jsonl",
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )
    item_result = run_item_mode(
        tasks,
        output_path=tmp_path / "item.jsonl",
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert [row.model_dump(mode="json") for row in pipeline_result] == [
        row.model_dump(mode="json") for row in item_result
    ]


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
