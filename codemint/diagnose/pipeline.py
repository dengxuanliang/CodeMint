from __future__ import annotations

import json
from pathlib import Path

from codemint.config import CodeMintConfig
from codemint.io.jsonl import append_jsonl, read_jsonl
from codemint.modeling.client import ModelClient
from codemint.modeling.parser import _normalize_json_text, parse_with_retry
from codemint.modeling.token_budget import truncate_payload
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord, SUCCESS_LIKE_SUB_TAGS
from codemint.models.task import TaskRecord
from codemint.prompts.registry import load_prompt
from codemint.rules.builtin import DiagnosisRule
from codemint.rules.custom import build_rules
from codemint.rules.engine import RuleEngine

from codemint.diagnose.confirm import (
    ConfirmAnalyzer,
    confirm_rule_with_model,
    default_confirm_analyzer,
)
from codemint.diagnose.deep import DeepAnalyzer, deep_analyze_with_model
from codemint.diagnose.resume import find_missing_task_ids


def run_diagnose(
    tasks: list[TaskRecord],
    output_path: Path,
    rules: list[DiagnosisRule] | None = None,
    *,
    confirm_analyzer: ConfirmAnalyzer | None = None,
    deep_analyzer: DeepAnalyzer | None = None,
    max_input_tokens: int = 8000,
    progress_callback=None,
    config: CodeMintConfig | None = None,
) -> list[DiagnosisRecord]:
    resolved_config = config or CodeMintConfig()
    active_rules = rules or build_rules()
    client = _build_model_client(resolved_config)
    confirmer = confirm_analyzer or _default_confirm_analyzer(client)
    deep = deep_analyzer or _default_deep_analyzer(client)
    engine = RuleEngine(active_rules)
    _validate_unique_task_ids(tasks)
    existing_diagnoses = _load_existing_diagnoses(output_path)

    missing_task_ids = set(find_missing_task_ids(output_path, [task.task_id for task in tasks]))
    new_diagnoses: list[DiagnosisRecord] = []
    for task in tasks:
        if task.task_id not in missing_task_ids:
            continue
        diagnosis = _diagnose_task_or_log_failure(
            task,
            output_path=output_path,
            engine=engine,
            confirm_analyzer=confirmer,
            deep_analyzer=deep,
            max_input_tokens=max_input_tokens,
        )
        if diagnosis is not None:
            new_diagnoses.append(diagnosis)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "diagnose",
                    "status": "running",
                    "processed": len(new_diagnoses),
                    "total": len(missing_task_ids),
                    "errors": 0,
                    "eta_seconds": max(len(missing_task_ids) - len(new_diagnoses), 0) * 3,
                }
            )

    if new_diagnoses:
        append_jsonl(output_path, [diagnosis.model_dump(mode="json") for diagnosis in new_diagnoses])
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=True)

    return existing_diagnoses + new_diagnoses


def _diagnose_task_or_log_failure(
    task: TaskRecord,
    *,
    output_path: Path,
    engine: RuleEngine,
    confirm_analyzer: ConfirmAnalyzer,
    deep_analyzer: DeepAnalyzer,
    max_input_tokens: int,
) -> DiagnosisRecord | None:
    try:
        return _diagnose_task(task, engine, confirm_analyzer, deep_analyzer, max_input_tokens)
    except Exception as error:
        append_jsonl(
            output_path.parent / "errors.jsonl",
            [
                {
                    "stage": "diagnose",
                    "task_id": task.task_id,
                    "error_type": "diagnosis_failed",
                    "message": str(error),
                }
            ],
        )
        return None


def _diagnose_task(
    task: TaskRecord,
    engine: RuleEngine,
    confirm_analyzer: ConfirmAnalyzer,
    deep_analyzer: DeepAnalyzer,
    max_input_tokens: int,
) -> DiagnosisRecord:
    matched_rule = engine.match(_task_text(task))
    budgeted_task = truncate_payload(task, max_input_tokens)

    if matched_rule and matched_rule.severity in {"medium", "high"}:
        if matched_rule.rule_id == "R010":
            return deep_analyze_with_model(budgeted_task, deep_analyzer)
        return confirm_rule_with_model(budgeted_task, matched_rule, confirm_analyzer)
    if matched_rule and matched_rule.rule_id != "R010":
        return diagnosis_from_rule_only(task, matched_rule)
    return deep_analyze_with_model(budgeted_task, deep_analyzer)


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


def _default_deep_analyzer(client: ModelClient | None) -> DeepAnalyzer:
    if client is None:
        def fallback(task: TaskRecord) -> DiagnosisRecord:
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

        return fallback

    prompt = load_prompt("diagnose_deep_analysis")

    def analyze(task: TaskRecord) -> DiagnosisRecord:
        payload = {
            "task_id": task.task_id,
            "content": task.content,
            "completion": task.completion,
            "test_code": task.test_code,
            "canonical_solution": task.canonical_solution,
            "labels": task.labels,
        }

        def invoke(format_error: str | None) -> str:
            user_prompt = f"{prompt.template}\n\nTask JSON:\n{payload}"
            if format_error:
                user_prompt += f"\n\nFormat correction:\n{format_error}"
            return client.complete("Return only valid JSON.", str(user_prompt))
        raw = invoke(None)
        try:
            return _parse_normalized_diagnosis(raw)
        except Exception as first_error:
            raw_retry = invoke(
                f"Return JSON using only allowed enum values for fault_type, severity, diagnosis_source. Error: {first_error}"
            )
            return _parse_normalized_diagnosis(raw_retry)

    return analyze


def _default_confirm_analyzer(client: ModelClient | None) -> ConfirmAnalyzer:
    if client is None:
        return default_confirm_analyzer

    prompt = load_prompt("diagnose_rule_confirm")

    def analyze(task: TaskRecord, rule: DiagnosisRule) -> DiagnosisRecord:
        payload = {
            "task_id": task.task_id,
            "rule_id": rule.rule_id,
            "fault_type": rule.fault_type,
            "sub_tag": rule.sub_tag,
            "severity": rule.severity,
            "content": task.content,
            "completion": task.completion,
            "test_code": task.test_code,
        }

        def invoke(format_error: str | None) -> str:
            user_prompt = f"{prompt.template}\n\nTask JSON:\n{payload}"
            if format_error:
                user_prompt += f"\n\nFormat correction:\n{format_error}"
            return client.complete("Return only valid JSON.", str(user_prompt))
        raw = invoke(None)
        try:
            return _parse_normalized_diagnosis(raw)
        except Exception as first_error:
            raw_retry = invoke(
                f"Return JSON using only allowed enum values for fault_type, severity, diagnosis_source. Error: {first_error}"
            )
            return _parse_normalized_diagnosis(raw_retry)

    return analyze


def _build_model_client(config: CodeMintConfig) -> ModelClient | None:
    model = config.model
    if not (model.analysis_model and model.base_url and model.api_key):
        return None
    return ModelClient(model)


def _parse_normalized_diagnosis(raw: str) -> DiagnosisRecord:
    payload = json.loads(_normalize_json_text(raw))
    normalized = _normalize_diagnosis_payload(payload)
    return DiagnosisRecord.model_validate(normalized)


def _normalize_diagnosis_payload(payload: dict) -> dict:
    normalized = dict(payload)
    normalized_sub_tags = _normalize_sub_tags(payload.get("sub_tags"))
    normalized["sub_tags"] = normalized_sub_tags
    normalized["fault_type"] = _normalize_fault_type(str(payload.get("fault_type", "")), normalized_sub_tags)
    normalized["severity"] = _normalize_severity(str(payload.get("severity", "")))
    normalized["evidence"] = _normalize_evidence(payload.get("evidence"), payload)
    normalized["enriched_labels"] = {
        str(key): str(value) for key, value in dict(payload.get("enriched_labels", {})).items()
    }
    normalized["diagnosis_source"] = _normalize_diagnosis_source(
        str(payload.get("diagnosis_source", "")),
        normalized_sub_tags,
        normalized["enriched_labels"],
        normalized["evidence"],
    )
    return normalized


def _normalize_sub_tags(value) -> list[str]:
    if isinstance(value, list):
        raw_tags = value
    elif value in (None, ""):
        raw_tags = []
    else:
        raw_tags = [value]

    normalized: list[str] = []
    for raw_tag in raw_tags:
        for canonical_tag in _canonicalize_sub_tag(str(raw_tag)):
            if canonical_tag not in normalized:
                normalized.append(canonical_tag)
    failure_tags = [tag for tag in normalized if tag not in SUCCESS_LIKE_SUB_TAGS]
    if failure_tags:
        return failure_tags
    return normalized or ["unknown"]


def _canonicalize_sub_tag(value: str) -> list[str]:
    candidate = value.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "public_entry_point_mismatch": ["function_name_mismatch"],
        "entry_point_mismatch": ["function_name_mismatch"],
        "function_name_non_compliance": ["function_name_mismatch"],
        "function_signature_mismatch": ["function_name_mismatch"],
        "wrong_function_name": ["function_name_mismatch"],
        "markdown_code_fence": ["markdown_formatting"],
        "code_fence_formatting": ["markdown_formatting"],
        "raw_output_formatting_violation": ["markdown_formatting"],
        "extraneous_characters": ["markdown_formatting"],
        "missing_code": ["missing_code_block"],
        "missing_output": ["missing_code_block"],
        "no_code_provided": ["missing_code_block"],
        "explanation_without_code": ["missing_code_block"],
        "commentary_instead_of_code": ["missing_code_block"],
        "correct_output": ["correct_output"],
        "correct_execution": ["correct_execution"],
        "correct_solution": ["correct_solution"],
        "pass": ["pass"],
    }
    return mapping.get(candidate, [candidate] if candidate else [])


def _normalize_fault_type(value: str, sub_tags: list[str]) -> str:
    if "function_name_mismatch" in sub_tags:
        return "surface"
    if "markdown_formatting" in sub_tags:
        return "surface"
    if "missing_code_block" in sub_tags:
        return "implementation"
    if _is_non_failure_sub_tags(sub_tags):
        return "implementation"
    candidate = value.strip().lower()
    mapping = {
        "syntaxerror": "surface",
        "missing output": "surface",
        "interface mismatch": "implementation",
        "logic error": "implementation",
        "runtime error": "implementation",
    }
    return mapping.get(candidate, candidate if candidate in {"comprehension", "modeling", "implementation", "edge_handling", "surface"} else "implementation")


def _normalize_severity(value: str) -> str:
    candidate = value.strip().lower()
    if candidate in {"critical", "fatal"}:
        return "high"
    if candidate in {"minor"}:
        return "low"
    return candidate if candidate in {"low", "medium", "high"} else "medium"


def _normalize_diagnosis_source(
    value: str,
    sub_tags: list[str],
    enriched_labels: dict[str, str],
    evidence: dict,
) -> str:
    candidate = value.strip().lower()
    mapping = {
        "static_analysis": "model_deep",
        "rule_based_analysis": "rule_confirmed_by_model",
        "deep_analysis": "model_deep",
        "non_failure": "non_failure",
        "correct_solution": "non_failure",
        "correct_execution": "non_failure",
        "correct_output": "non_failure",
        "pass_through": "non_failure",
        "pass": "non_failure",
    }
    normalized = mapping.get(
        candidate,
        candidate if candidate in {"rule_confirmed_by_model", "rule_only", "model_deep", "non_failure"} else "model_deep",
    )
    if _should_mark_non_failure(sub_tags, enriched_labels, evidence):
        return "non_failure"
    if normalized == "non_failure":
        return "model_deep"
    return normalized


def _should_mark_non_failure(
    sub_tags: list[str],
    enriched_labels: dict[str, str],
    evidence: dict,
) -> bool:
    if any(tag in {"function_name_mismatch", "markdown_formatting", "missing_code_block"} for tag in sub_tags):
        return False
    if _is_non_failure_sub_tags(sub_tags):
        return True

    normalized_labels = {
        key.strip().lower(): value.strip().lower()
        for key, value in enriched_labels.items()
    }
    if normalized_labels.get("status") == "correct_solution":
        return True
    if normalized_labels.get("test_result") == "pass":
        return True
    if normalized_labels.get("functional_correctness") == "true":
        return True

    wrong_line = str(evidence.get("wrong_line", "")).strip().upper()
    failed_test = str(evidence.get("failed_test", "")).strip().upper()
    return wrong_line == "N/A" and failed_test == "N/A"


def _is_non_failure_sub_tags(sub_tags: list[str]) -> bool:
    normalized = {tag.strip().lower() for tag in sub_tags}
    return bool(normalized) and normalized.issubset(SUCCESS_LIKE_SUB_TAGS)


def _normalize_evidence(value, payload: dict) -> dict:
    if isinstance(value, dict):
        return {
            "wrong_line": str(value.get("wrong_line") or value.get("completion_snippet") or payload.get("completion", "")),
            "correct_approach": str(value.get("correct_approach") or value.get("fix") or "Inspect the failing behavior in context and reason from tests."),
            "failed_test": str(value.get("failed_test") or value.get("error_message") or payload.get("test_code", "")),
        }
    text = str(value or payload.get("completion", ""))
    return {
        "wrong_line": text,
        "correct_approach": "Inspect the failing behavior in context and reason from tests.",
        "failed_test": str(payload.get("test_code", "")),
    }


def _validate_unique_task_ids(tasks: list[TaskRecord]) -> None:
    seen: set[int] = set()
    for task in tasks:
        if task.task_id in seen:
            raise ValueError(f"Duplicate task_id in input tasks: {task.task_id}")
        seen.add(task.task_id)


def _load_existing_diagnoses(output_path: Path) -> list[DiagnosisRecord]:
    if not output_path.exists():
        return []

    return [DiagnosisRecord.model_validate(row) for row in read_jsonl(output_path)]
