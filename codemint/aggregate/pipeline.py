from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from codemint.aggregate.cluster import cluster_diagnoses
from codemint.aggregate.causal import CausalAnalyzer, build_causal_chains
from codemint.aggregate.collective import (
    CollectiveAnalysisResult,
    CollectiveAnalyzer,
    CollectiveCluster,
    apply_collective_diagnosis,
    default_collective_analyze,
)
from codemint.config import CodeMintConfig
from codemint.io.jsonl import append_jsonl
from codemint.aggregate.rank import build_rankings
from codemint.aggregate.repair import (
    VerificationLevel,
    VerificationResult,
    default_verify,
    repair_diagnosis,
)
from codemint.modeling.client import ModelClient
from codemint.modeling.parser import _normalize_json_text
from codemint.models.diagnosis import DiagnosisRecord, FaultType
from codemint.models.weakness import (
    CausalChain,
    CollectiveDiagnosis,
    RankingSet,
    WeaknessEntry,
    WeaknessReport,
)
from codemint.prompts.registry import load_prompt


_ALLOWED_CANONICAL_WEAKNESSES = frozenset(
    {
        "function_name_mismatch",
        "markdown_formatting",
        "missing_code_block",
        "syntax_error",
        "logic_error",
        "non_executable_code",
    }
)

_CANONICAL_WEAKNESS_ALIASES = {
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
    "off_by_one": "logic_error",
    "index_bounds": "logic_error",
    "indexing": "logic_error",
    "loop_bound": "logic_error",
    "boundary_error": "logic_error",
}


def run_aggregate(
    diagnoses: list[DiagnosisRecord],
    output_path: Path,
    *,
    config: CodeMintConfig | None = None,
    verification_level: VerificationLevel = "auto",
    verify: Callable[[DiagnosisRecord, VerificationLevel], dict | VerificationResult] | None = None,
    rediagnose: Callable[[DiagnosisRecord], DiagnosisRecord] | None = None,
    collective_analyze: CollectiveAnalyzer | None = None,
    causal_analyze: CausalAnalyzer | None = None,
    progress_callback=None,
) -> WeaknessReport:
    verifier = verify or default_verify
    rediagnoser = rediagnose or _identity
    resolved_config = config or CodeMintConfig()
    failure_diagnoses = [diagnosis for diagnosis in diagnoses if _is_failure_diagnosis(diagnosis)]
    repaired = [
        repair_diagnosis(
            diagnosis,
            verification_level=verification_level,
            verify=verifier,
            rediagnose=rediagnoser,
        )
        for diagnosis in failure_diagnoses
    ]
    clusters = cluster_diagnoses(repaired)
    collective_clusters, tag_mappings = apply_collective_diagnosis(
        clusters,
        _build_resilient_collective_analyzer(
            output_path=output_path,
            analyze=collective_analyze or _default_collective_analyzer(resolved_config),
        ),
    )
    tag_mappings = _normalize_tag_mappings(tag_mappings)
    normalized = _apply_collective_adjustments(repaired, collective_clusters, tag_mappings)
    final_clusters = cluster_diagnoses(normalized)
    if progress_callback is not None:
        for index, _cluster in enumerate(final_clusters, start=1):
            progress_callback(
                {
                    "stage": "aggregate",
                    "status": "running",
                    "processed": index,
                    "total": len(final_clusters),
                    "errors": 0,
                    "eta_seconds": max(len(final_clusters) - index, 0) * 3,
                }
            )
    report = _build_report(final_clusters, collective_clusters, tag_mappings, causal_analyze)
    _write_report(output_path, report)
    return report


def _build_report(
    clusters,
    collective_clusters: list[CollectiveCluster],
    tag_mappings: dict[str, str],
    causal_analyze: CausalAnalyzer | None,
) -> WeaknessReport:
    weaknesses: list[WeaknessEntry] = []
    collective_by_key = {
        (cluster.fault_type, _canonicalize_sub_tag(cluster.sub_tags[0] if cluster.sub_tags else "unknown")):
        cluster.collective_diagnosis
        for cluster in collective_clusters
    }
    for index, cluster in enumerate(clusters, start=1):
        weaknesses.append(
            WeaknessEntry(
                rank=index,
                fault_type=cluster.fault_type,
                sub_tags=cluster.sub_tags,
                frequency=len(cluster.diagnoses),
                sample_task_ids=cluster.task_ids[:3],
                trainability=_trainability_for_fault_type(cluster.fault_type),
                collective_diagnosis=collective_by_key.get(cluster.key)
                or _default_collective_diagnosis(cluster),
            )
        )

    return WeaknessReport(
        weaknesses=weaknesses,
        rankings=build_rankings(weaknesses),
        causal_chains=build_causal_chains(weaknesses, causal_analyze),
        tag_mappings=tag_mappings,
    )


def _write_report(output_path: Path, report: WeaknessReport) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _identity(diagnosis: DiagnosisRecord) -> DiagnosisRecord:
    return diagnosis.model_copy(deep=True)


def _is_failure_diagnosis(diagnosis: DiagnosisRecord) -> bool:
    return diagnosis.is_failure


def _apply_collective_adjustments(
    diagnoses: list[DiagnosisRecord],
    collective_clusters: list[CollectiveCluster],
    tag_mappings: dict[str, str],
) -> list[DiagnosisRecord]:
    adjusted: list[DiagnosisRecord] = []

    for diagnosis in diagnoses:
        current = diagnosis.model_copy(deep=True)
        current.sub_tags = _normalize_sub_tags(current.sub_tags, tag_mappings)

        adjusted.append(current)

    return adjusted


def _normalize_sub_tags(sub_tags: list[str], tag_mappings: dict[str, str]) -> list[str]:
    normalized: list[str] = []
    for tag in sub_tags:
        mapped = _canonicalize_sub_tag(tag_mappings.get(tag, tag))
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized or ["unknown"]


def _normalize_tag_mappings(tag_mappings: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for source_tag, target_tag in tag_mappings.items():
        normalized[source_tag] = _canonicalize_sub_tag(target_tag)
    return normalized


def _canonicalize_sub_tag(tag: str) -> str:
    normalized = tag.strip()
    canonical = _CANONICAL_WEAKNESS_ALIASES.get(normalized, normalized)
    if canonical in _ALLOWED_CANONICAL_WEAKNESSES:
        return canonical
    return "logic_error"


def _default_collective_diagnosis(cluster) -> CollectiveDiagnosis:
    verification_levels = sorted(
        {
            diagnosis.enriched_labels.get("verification_level")
            for diagnosis in cluster.diagnoses
            if diagnosis.enriched_labels.get("verification_level")
        }
    )
    verification_statuses = sorted(
        {
            diagnosis.enriched_labels.get("verification_status")
            for diagnosis in cluster.diagnoses
            if diagnosis.enriched_labels.get("verification_status")
        }
    )
    primary_tag = cluster.sub_tags[0] if cluster.sub_tags else "unknown"
    return CollectiveDiagnosis(
        refined_root_cause=(
            f"Grouped by {cluster.fault_type}/{primary_tag} "
            f"(verification_status={','.join(verification_statuses) or 'unknown'}; "
            f"verification_level={','.join(verification_levels) or 'unknown'})"
        ),
        capability_cliff=f"{primary_tag} remains after canonical clustering.",
        misdiagnosed_ids=[],
        misdiagnosis_corrections={},
        cluster_coherence=1.0,
    )


def _trainability_for_fault_type(fault_type: str) -> float:
    trainability_by_fault_type = {
        "modeling": 0.9,
        "comprehension": 0.75,
        "implementation": 0.6,
        "edge_handling": 0.5,
        "surface": 0.3,
    }
    return trainability_by_fault_type.get(fault_type, 0.5)


def _build_resilient_collective_analyzer(
    *,
    output_path: Path,
    analyze: CollectiveAnalyzer | None,
) -> CollectiveAnalyzer:
    analyzer = analyze or default_collective_analyze
    cache: dict[str, dict] = {}

    def resilient(payload: dict) -> dict:
        cache_key = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if cache_key in cache:
            return cache[cache_key]

        last_error: ValidationError | None = None
        for attempt in range(1, 3):
            candidate = analyzer(payload)
            try:
                parsed = CollectiveAnalysisResult.model_validate(candidate)
            except ValidationError as error:
                last_error = error
                if attempt == 2:
                    fallback = default_collective_analyze(payload)
                    _log_collective_parse_failure(
                        errors_path=output_path.parent / "errors.jsonl",
                        payload=payload,
                        attempts=attempt,
                        error=error,
                    )
                    cache[cache_key] = fallback
                    return fallback
                continue

            normalized = parsed.model_dump(mode="json")
            cache[cache_key] = normalized
            return normalized

        assert last_error is not None
        raise last_error

    return resilient


def _default_collective_analyzer(config: CodeMintConfig) -> CollectiveAnalyzer:
    client = _build_model_client(config)
    if client is None:
        return default_collective_analyze

    prompt = load_prompt("aggregate_collective_diagnosis")

    def analyze(payload: dict) -> dict:
        def invoke(format_error: str | None) -> str:
            user_prompt = f"{prompt.template}\n\nPayload JSON:\n{payload}"
            if format_error:
                user_prompt += f"\n\nFormat correction:\n{format_error}"
            return client.complete("Return only valid JSON.", str(user_prompt))

        raw = invoke(None)
        try:
            return _parse_normalized_collective_analysis(raw)
        except Exception as first_error:
            raw_retry = invoke(f"Return JSON matching the required schema exactly. Error: {first_error}")
            return _parse_normalized_collective_analysis(raw_retry)

    return analyze


def _build_model_client(config: CodeMintConfig) -> ModelClient | None:
    model = config.model
    if not (model.analysis_model and model.base_url and model.api_key):
        return None
    return ModelClient(model)


def _parse_normalized_collective_analysis(raw: str) -> dict:
    payload = json.loads(_normalize_json_text(raw))
    return _normalize_collective_payload(payload)


def _normalize_collective_payload(payload: dict) -> dict:
    normalized = dict(payload)
    normalized["misdiagnosed_ids"] = _normalize_int_list(payload.get("misdiagnosed_ids"))
    normalized["misdiagnosis_corrections"] = {
        str(key): str(value) for key, value in dict(payload.get("misdiagnosis_corrections", {})).items()
    }
    normalized["cluster_coherence"] = float(payload.get("cluster_coherence", 0.0))
    normalized["semantic_merges"] = [
        {
            "source_tag": str(item.get("source_tag", "")),
            "target_tag": str(item.get("target_tag", "")),
            "confirmed": _normalize_bool(item.get("confirmed", False)),
        }
        for item in list(payload.get("semantic_merges", []))
        if isinstance(item, dict)
    ]
    return normalized


def _normalize_int_list(value) -> list[int]:
    normalized: list[int] = []
    for item in list(value or []):
        try:
            normalized.append(int(item))
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _log_collective_parse_failure(
    *,
    errors_path: Path,
    payload: dict,
    attempts: int,
    error: ValidationError,
) -> None:
    append_jsonl(
        errors_path,
        [
            {
                "stage": "aggregate",
                "error_type": "collective_parse_failure",
                "attempts": attempts,
                "cluster": {
                    "fault_type": payload["cluster"]["fault_type"],
                    "sub_tags": payload["cluster"]["sub_tags"],
                    "task_ids": payload["cluster"]["task_ids"],
                },
                "message": str(error),
            }
        ],
    )
