from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from codemint.diagnose.confirm import ConfirmAnalyzer, confirm_rule_with_model
from codemint.diagnose.deep import DeepAnalyzer, deep_analyze_with_model
from codemint.diagnose.resume import find_missing_task_ids
from codemint.io.jsonl import append_jsonl, read_jsonl
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.builtin import DiagnosisRule
from codemint.rules.custom import build_rules
from codemint.rules.engine import RuleEngine


def run_diagnose(
    tasks: list[TaskRecord],
    output_path: Path,
    rules: list[DiagnosisRule] | None = None,
    *,
    confirm_analyzer: ConfirmAnalyzer | None = None,
    deep_analyzer: DeepAnalyzer | None = None,
) -> list[DiagnosisRecord]:
    active_rules = rules or build_rules()
    confirmer = confirm_analyzer or _default_confirm_analyzer
    deep = deep_analyzer or _default_deep_analyzer
    engine = RuleEngine(active_rules)

    missing_task_ids = set(find_missing_task_ids(output_path, [task.task_id for task in tasks]))
    new_diagnoses: list[DiagnosisRecord] = []
    for task in tasks:
        if task.task_id not in missing_task_ids:
            continue
        new_diagnoses.append(_diagnose_task(task, engine, confirmer, deep))

    if new_diagnoses:
        append_jsonl(output_path, [diagnosis.model_dump(mode="json") for diagnosis in new_diagnoses])

    all_rows = read_jsonl(output_path) if output_path.exists() else []
    return [DiagnosisRecord.model_validate(row) for row in all_rows]


def _diagnose_task(
    task: TaskRecord,
    engine: RuleEngine,
    confirm_analyzer: ConfirmAnalyzer,
    deep_analyzer: DeepAnalyzer,
) -> DiagnosisRecord:
    matched_rule = engine.match(_task_text(task))

    if matched_rule and matched_rule.severity in {"medium", "high"}:
        if matched_rule.rule_id == "R010":
            return deep_analyze_with_model(task, deep_analyzer)
        return confirm_rule_with_model(task, matched_rule, confirm_analyzer)
    if matched_rule and matched_rule.rule_id != "R010":
        return diagnosis_from_rule_only(task, matched_rule)
    return deep_analyze_with_model(task, deep_analyzer)


def diagnosis_from_rule_only(task: TaskRecord, rule: DiagnosisRule) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task.task_id,
        fault_type=rule.fault_type,
        sub_tags=[rule.sub_tag],
        severity=rule.severity,
        description=f"Matched diagnosis rule {rule.rule_id}.",
        evidence=DiagnosisEvidence(
            wrong_line=task.completion,
            correct_approach=f"Investigate the issue classified by {rule.sub_tag}.",
            failed_test=task.test_code,
        ),
        enriched_labels={},
        confidence=0.6,
        diagnosis_source="rule_only",
        prompt_version="rule-only-v1",
    )


def _task_text(task: TaskRecord) -> str:
    parts = [task.content, task.completion, task.test_code]
    return "\n".join(part for part in parts if part)


def _default_confirm_analyzer(task: TaskRecord, rule_id: str) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task.task_id,
        fault_type="implementation",
        sub_tags=[f"confirmed_{rule_id.lower()}"],
        severity="medium",
        description=f"Model-confirmed diagnosis for {rule_id}.",
        evidence=DiagnosisEvidence(
            wrong_line=task.completion,
            correct_approach="Review the matched rule context and repair the implementation.",
            failed_test=task.test_code,
        ),
        enriched_labels={"rule_id": rule_id},
        confidence=0.75,
        diagnosis_source="rule_confirmed_by_model",
        prompt_version="confirm-v1",
    )


def _default_deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task.task_id,
        fault_type="implementation",
        sub_tags=["deep_analysis"],
        severity="medium",
        description="Deep model analysis required.",
        evidence=DiagnosisEvidence(
            wrong_line=task.completion,
            correct_approach="Inspect the failing behavior in context and reason from tests.",
            failed_test=task.test_code,
        ),
        enriched_labels={},
        confidence=0.7,
        diagnosis_source="model_deep",
        prompt_version="deep-v1",
    )
