from __future__ import annotations

from pathlib import Path

from codemint.diagnose.confirm import (
    ConfirmAnalyzer,
    confirm_rule_with_model,
    default_confirm_analyzer,
)
from codemint.diagnose.deep import DeepAnalyzer, deep_analyze_with_model
from codemint.diagnose.resume import find_missing_task_ids
from codemint.io.jsonl import append_jsonl, read_jsonl
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord, SUCCESS_LIKE_SUB_TAGS
from codemint.models.task import TaskRecord
from codemint.rules.builtin import DiagnosisRule
from codemint.rules.custom import build_rules
from codemint.rules.engine import RuleEngine

_CANONICAL_SUB_TAGS = {
    "public_entry_point_mismatch": "function_name_mismatch",
    "entry_point_mismatch": "function_name_mismatch",
    "function_name_non_compliance": "function_name_mismatch",
    "function_signature_mismatch": "function_name_mismatch",
    "wrong_function_name": "function_name_mismatch",
    "markdown_code_fence": "markdown_formatting",
    "code_fence_formatting": "markdown_formatting",
    "raw_output_formatting_violation": "markdown_formatting",
    "extraneous_characters": "markdown_formatting",
    "missing_code": "missing_code_block",
    "missing_output": "missing_code_block",
    "no_code_provided": "missing_code_block",
    "explanation_without_code": "missing_code_block",
    "commentary_instead_of_code": "missing_code_block",
}


def run_diagnose(
    tasks: list[TaskRecord],
    output_path: Path,
    rules: list[DiagnosisRule] | None = None,
    *,
    confirm_analyzer: ConfirmAnalyzer | None = None,
    deep_analyzer: DeepAnalyzer | None = None,
) -> list[DiagnosisRecord]:
    active_rules = rules or build_rules()
    confirmer = confirm_analyzer or default_confirm_analyzer
    deep = deep_analyzer or _default_deep_analyzer
    engine = RuleEngine(active_rules)
    _validate_unique_task_ids(tasks)
    existing_diagnoses = _load_existing_diagnoses(output_path)

    missing_task_ids = set(find_missing_task_ids(output_path, [task.task_id for task in tasks]))
    new_diagnoses: list[DiagnosisRecord] = []
    for task in tasks:
        if task.task_id not in missing_task_ids:
            continue
        new_diagnoses.append(_diagnose_task(task, engine, confirmer, deep))

    if new_diagnoses:
        append_jsonl(output_path, [diagnosis.model_dump(mode="json") for diagnosis in new_diagnoses])

    return existing_diagnoses + new_diagnoses


def _diagnose_task(
    task: TaskRecord,
    engine: RuleEngine,
    confirm_analyzer: ConfirmAnalyzer,
    deep_analyzer: DeepAnalyzer,
) -> DiagnosisRecord:
    matched_rule = engine.match(_task_text(task))

    if matched_rule and matched_rule.severity in {"medium", "high"}:
        if matched_rule.rule_id == "R010":
            return _normalize_diagnosis_record(deep_analyze_with_model(task, deep_analyzer))
        return _normalize_diagnosis_record(confirm_rule_with_model(task, matched_rule, confirm_analyzer))
    if matched_rule and matched_rule.rule_id != "R010":
        return _normalize_diagnosis_record(diagnosis_from_rule_only(task, matched_rule))
    return _normalize_diagnosis_record(deep_analyze_with_model(task, deep_analyzer))


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


def _normalize_diagnosis_record(record: DiagnosisRecord) -> DiagnosisRecord:
    normalized_sub_tags = _normalize_sub_tags(record.sub_tags)
    diagnosis_source = _normalize_diagnosis_source(record.diagnosis_source, normalized_sub_tags)
    return record.model_copy(
        update={
            "sub_tags": normalized_sub_tags,
            "diagnosis_source": diagnosis_source,
        }
    )


def _normalize_sub_tags(sub_tags: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw_tag in sub_tags:
        candidate = str(raw_tag).strip().lower().replace("-", "_").replace(" ", "_")
        if not candidate:
            continue
        canonical = _CANONICAL_SUB_TAGS.get(candidate, candidate)
        if canonical not in normalized:
            normalized.append(canonical)

    failure_tags = [tag for tag in normalized if tag not in SUCCESS_LIKE_SUB_TAGS]
    if failure_tags:
        return failure_tags
    return normalized or ["unknown"]


def _normalize_diagnosis_source(diagnosis_source: str, sub_tags: list[str]) -> str:
    if sub_tags and set(sub_tags).issubset(SUCCESS_LIKE_SUB_TAGS):
        return "non_failure"
    return diagnosis_source


def _validate_unique_task_ids(tasks: list[TaskRecord]) -> None:
    seen: set[int] = set()
    for task in tasks:
        if task.task_id in seen:
            raise ValueError(f"Duplicate task_id in input tasks: {task.task_id}")
        seen.add(task.task_id)


def _load_existing_diagnoses(output_path: Path) -> list[DiagnosisRecord]:
    if not output_path.exists():
        return []

    return [_normalize_diagnosis_record(DiagnosisRecord.model_validate(row)) for row in read_jsonl(output_path)]
