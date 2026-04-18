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
from codemint.io.jsonl import append_jsonl
from codemint.aggregate.rank import build_rankings
from codemint.aggregate.repair import (
    VerificationLevel,
    VerificationResult,
    default_verify,
    repair_diagnosis,
)
from codemint.models.diagnosis import DiagnosisRecord, FaultType
from codemint.models.weakness import (
    CausalChain,
    CollectiveDiagnosis,
    RankingSet,
    WeaknessEntry,
    WeaknessReport,
)


def run_aggregate(
    diagnoses: list[DiagnosisRecord],
    output_path: Path,
    *,
    verification_level: VerificationLevel = "auto",
    verify: Callable[[DiagnosisRecord, VerificationLevel], dict | VerificationResult] | None = None,
    rediagnose: Callable[[DiagnosisRecord], DiagnosisRecord] | None = None,
    collective_analyze: CollectiveAnalyzer | None = None,
    causal_analyze: CausalAnalyzer | None = None,
) -> WeaknessReport:
    verifier = verify or default_verify
    rediagnoser = rediagnose or _identity
    repaired = [
        repair_diagnosis(
            diagnosis,
            verification_level=verification_level,
            verify=verifier,
            rediagnose=rediagnoser,
        )
        for diagnosis in diagnoses
    ]
    clusters = cluster_diagnoses(repaired)
    collective_clusters, tag_mappings = apply_collective_diagnosis(
        clusters,
        _build_resilient_collective_analyzer(
            output_path=output_path,
            analyze=collective_analyze,
        ),
    )
    normalized = _apply_collective_adjustments(repaired, collective_clusters, tag_mappings)
    final_clusters = cluster_diagnoses(normalized)
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
    collective_by_key = {cluster.key: cluster.collective_diagnosis for cluster in collective_clusters}
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


def _apply_collective_adjustments(
    diagnoses: list[DiagnosisRecord],
    collective_clusters: list[CollectiveCluster],
    tag_mappings: dict[str, str],
) -> list[DiagnosisRecord]:
    reclassifications = _build_reclassifications(collective_clusters)
    adjusted: list[DiagnosisRecord] = []

    for diagnosis in diagnoses:
        current = diagnosis.model_copy(deep=True)
        current.sub_tags = _normalize_sub_tags(current.sub_tags, tag_mappings)

        correction = reclassifications.get(current.task_id)
        if correction is not None:
            current.fault_type = correction[0]
            current.sub_tags = _normalize_sub_tags([correction[1]], tag_mappings)

        adjusted.append(current)

    return adjusted


def _build_reclassifications(
    collective_clusters: list[CollectiveCluster],
) -> dict[int, tuple[str, str]]:
    reclassifications: dict[int, tuple[str, str]] = {}
    for cluster in collective_clusters:
        for task_id_text, correction in cluster.collective_diagnosis.misdiagnosis_corrections.items():
            task_id = _parse_task_id(task_id_text)
            parsed = _parse_correction(correction)
            if task_id is None or parsed is None:
                continue
            if not _is_valid_fault_type(parsed[0]):
                continue
            reclassifications[task_id] = parsed
    return reclassifications


def _parse_correction(value: str) -> tuple[str, str] | None:
    fault_type, separator, sub_tag = value.partition(":")
    if not separator or not sub_tag:
        return None
    return fault_type, sub_tag


def _parse_task_id(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_valid_fault_type(value: str) -> bool:
    return value in {
        "comprehension",
        "modeling",
        "implementation",
        "edge_handling",
        "surface",
    }


def _normalize_sub_tags(sub_tags: list[str], tag_mappings: dict[str, str]) -> list[str]:
    normalized: list[str] = []
    for tag in sub_tags:
        mapped = tag_mappings.get(tag, tag)
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized or ["unknown"]


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
        capability_cliff=f"{primary_tag} emerges after collective reclassification.",
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
