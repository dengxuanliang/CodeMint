from __future__ import annotations

import json
from pathlib import Path

from codemint.config import CodeMintConfig
from codemint.diagnose.confirm import (
    ConfirmAnalyzer,
    confirm_rule_with_model,
    default_confirm_analyzer,
)
from codemint.diagnose.deep import DeepAnalyzer, deep_analyze_with_model
from codemint.diagnose.resume import find_missing_task_ids
from codemint.io.jsonl import append_jsonl, read_jsonl
from codemint.modeling.client import ModelClient
from codemint.modeling.parser import parse_with_retry
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord, SUCCESS_LIKE_SUB_TAGS
from codemint.models.task import TaskRecord
from codemint.prompts.registry import load_prompt
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
    "missing_colon": "syntax_error",
}

_ALLOWED_FAILURE_SUB_TAGS = frozenset(
    {
        "function_name_mismatch",
        "markdown_formatting",
        "missing_code_block",
        "syntax_error",
        "logic_error",
        "non_executable_code",
    }
)


def run_item_mode(
    tasks: list[TaskRecord],
    output_path: Path,
    rules: list[DiagnosisRule] | None = None,
    *,
    config: CodeMintConfig | None = None,
    confirm_analyzer: ConfirmAnalyzer | None = None,
    deep_analyzer: DeepAnalyzer | None = None,
) -> list[DiagnosisRecord]:
    active_rules = rules or build_rules()
    resolved_config = config or CodeMintConfig()
    confirmer = confirm_analyzer or _default_confirm_analyzer(resolved_config)
    deep = deep_analyzer or _default_deep_analyzer(resolved_config)
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


def _default_deep_analyzer(config: CodeMintConfig) -> DeepAnalyzer:
    client = _build_model_client(config)
    if client is None:
        return _fallback_deep_analyzer

    prompt = load_prompt("diagnose_deep_analysis")

    def analyze(task: TaskRecord) -> DiagnosisRecord:
        payload = {
            "task_id": task.task_id,
            "content": task.content,
            "completion": task.completion,
            "test_code": task.test_code,
            "labels": task.labels,
            "accepted": task.accepted,
            "metrics": task.metrics,
        }

        def invoke(format_error: str | None) -> str:
            user_prompt = f"{prompt.template}\n\nPayload JSON:\n{json.dumps(payload, ensure_ascii=False)}"
            if format_error:
                user_prompt += f"\n\nFormat correction:\n{format_error}"
            return client.complete("Return only valid JSON.", str(user_prompt))

        record = _parse_diagnosis_with_retry(invoke)
        return record.model_copy(update={"prompt_version": prompt.version})

    return analyze


def _default_confirm_analyzer(config: CodeMintConfig) -> ConfirmAnalyzer:
    client = _build_model_client(config)
    if client is None:
        return default_confirm_analyzer

    prompt = load_prompt("diagnose_deep_analysis")

    def analyze(task: TaskRecord, rule: DiagnosisRule) -> DiagnosisRecord:
        payload = {
            "task_id": task.task_id,
            "content": task.content,
            "completion": task.completion,
            "test_code": task.test_code,
            "matched_rule": {
                "rule_id": rule.rule_id,
                "fault_type": rule.fault_type,
                "sub_tag": rule.sub_tag,
                "severity": rule.severity,
            },
        }

        def invoke(format_error: str | None) -> str:
            user_prompt = (
                f"{prompt.template}\n\nPayload JSON:\n{json.dumps(payload, ensure_ascii=False)}"
                "\n\nThe matched rule is a high-confidence hint, but you must still return the final diagnosis JSON."
            )
            if format_error:
                user_prompt += f"\n\nFormat correction:\n{format_error}"
            return client.complete("Return only valid JSON.", str(user_prompt))

        record = _parse_diagnosis_with_retry(invoke)
        return record.model_copy(
            update={
                "diagnosis_source": "rule_confirmed_by_model"
                if record.diagnosis_source != "non_failure"
                else "non_failure",
                "prompt_version": prompt.version,
            }
        )

    return analyze


def _fallback_deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task.task_id,
        fault_type="implementation",
        sub_tags=["deep_analysis"],
        severity="medium",
        description="Deep model analysis unavailable; fallback placeholder recorded.",
        evidence=DiagnosisEvidence(
            wrong_line=task.completion,
            correct_approach="Inspect the failing behavior in context and reason from tests.",
            failed_test=task.test_code,
        ),
        enriched_labels={"fallback_mode": "true"},
        confidence=0.2,
        diagnosis_source="model_deep",
        prompt_version="deep-fallback-v1",
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


def _build_model_client(config: CodeMintConfig) -> ModelClient | None:
    model = config.model
    if not (model.analysis_model and model.base_url and model.api_key):
        return None
    return ModelClient(model)


def _parse_diagnosis_with_retry(invoke) -> DiagnosisRecord:
    record = parse_with_retry(DiagnosisRecord, invoke)
    normalized = _normalize_diagnosis_record(record)
    error = _taxonomy_error(normalized)
    if error is None:
        return normalized

    reparsed = parse_with_retry(
        DiagnosisRecord,
        lambda _: invoke(
            f"{error} Use only the allowed taxonomy: "
            f"{sorted(_ALLOWED_FAILURE_SUB_TAGS)} plus success-like tags "
            f"{sorted(SUCCESS_LIKE_SUB_TAGS)}."
        ),
    )
    normalized_retry = _normalize_diagnosis_record(reparsed)
    error_retry = _taxonomy_error(normalized_retry)
    if error_retry is not None:
        raise ValueError(error_retry)
    return normalized_retry


def _taxonomy_error(record: DiagnosisRecord) -> str | None:
    if record.diagnosis_source == "non_failure":
        return None
    if not record.sub_tags:
        return "Diagnosis must include at least one sub_tag."

    primary_tag = record.sub_tags[0]
    if primary_tag not in _ALLOWED_FAILURE_SUB_TAGS:
        return f"Primary sub_tag {primary_tag!r} is outside the allowed taxonomy."

    illegal = [
        tag
        for tag in record.sub_tags
        if tag not in _ALLOWED_FAILURE_SUB_TAGS and tag not in SUCCESS_LIKE_SUB_TAGS
    ]
    if illegal:
        return f"Sub_tags {illegal!r} are outside the allowed taxonomy."
    return None
