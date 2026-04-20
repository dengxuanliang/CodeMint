from __future__ import annotations

from pathlib import Path

from codemint.diagnose import pipeline as pipeline_module
from codemint.diagnose.item_mode import run_item_mode
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.custom import build_rules


def test_item_mode_matches_legacy_pipeline_behavior(tmp_path: Path) -> None:
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

    legacy_result = _legacy_run_diagnose(
        tasks,
        tmp_path / "legacy.jsonl",
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

    assert [row.model_dump(mode="json") for row in legacy_result] == [
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


def _legacy_run_diagnose(
    tasks: list[TaskRecord],
    output_path: Path,
    *,
    rules,
    confirm_analyzer,
    deep_analyzer,
):
    active_rules = rules or pipeline_module.build_rules()
    confirmer = confirm_analyzer or pipeline_module._default_confirm_analyzer(pipeline_module.CodeMintConfig())
    deep = deep_analyzer or pipeline_module._default_deep_analyzer(pipeline_module.CodeMintConfig())
    engine = pipeline_module.RuleEngine(active_rules)
    pipeline_module._validate_unique_task_ids(tasks)
    existing_diagnoses = pipeline_module._load_existing_diagnoses(output_path)

    missing_task_ids = set(
        pipeline_module.find_missing_task_ids(output_path, [task.task_id for task in tasks])
    )
    new_diagnoses = []
    for task in tasks:
        if task.task_id not in missing_task_ids:
            continue
        new_diagnoses.append(pipeline_module._diagnose_task(task, engine, confirmer, deep))

    if new_diagnoses:
        pipeline_module.append_jsonl(
            output_path,
            [diagnosis.model_dump(mode="json") for diagnosis in new_diagnoses],
        )

    return existing_diagnoses + new_diagnoses
