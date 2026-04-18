from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import Field

from codemint.aggregate.cluster import DiagnosisCluster
from codemint.models.base import StrictModel
from codemint.models.diagnosis import DiagnosisRecord, FaultType
from codemint.models.weakness import CollectiveDiagnosis


CollectiveAnalyzer = Callable[[dict], dict]


class SemanticMerge(StrictModel):
    source_tag: str
    target_tag: str
    confirmed: bool


class CollectiveAnalysisResult(StrictModel):
    refined_root_cause: str
    capability_cliff: str
    misdiagnosed_ids: list[int]
    misdiagnosis_corrections: dict[str, str]
    cluster_coherence: float = Field(ge=0.0, le=1.0)
    semantic_merges: list[SemanticMerge] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CollectiveCluster:
    key: tuple[FaultType, str]
    fault_type: FaultType
    sub_tags: list[str]
    diagnoses: list[DiagnosisRecord]
    task_ids: list[int]
    collective_diagnosis: CollectiveDiagnosis


def apply_collective_diagnosis(
    clusters: list[DiagnosisCluster],
    analyze: CollectiveAnalyzer | None = None,
) -> tuple[list[CollectiveCluster], dict[str, str]]:
    analyzer = analyze or default_collective_analyze
    cluster_map = _cluster_map(clusters)
    ordered_keys = _ordered_cluster_keys(clusters)
    raw_tag_mappings = _initialize_tag_mappings(clusters)

    # Phase 1: collect confirmed merge edges without emitting output yet.
    for key in ordered_keys:
        cluster = cluster_map[key]
        primary_tag = cluster.sub_tags[0] if cluster.sub_tags else "unknown"
        resolved_primary_tag = _resolve_canonical_tag(primary_tag, raw_tag_mappings)
        if resolved_primary_tag != primary_tag:
            continue
        candidate_keys = [
            candidate_key
            for candidate_key in ordered_keys
            if candidate_key != key and candidate_key[0] == key[0]
        ]
        analysis = CollectiveAnalysisResult.model_validate(
            analyzer(
                _build_collective_payload(
                    cluster=cluster,
                    merged_diagnoses=cluster.diagnoses,
                    candidate_clusters=[cluster_map[candidate_key] for candidate_key in candidate_keys],
                    tag_mappings=_resolve_tag_mappings(raw_tag_mappings),
                )
            )
        )
        for merge in analysis.semantic_merges:
            if merge.confirmed:
                raw_tag_mappings[merge.source_tag] = merge.target_tag

    resolved_tag_mappings = _resolve_tag_mappings(raw_tag_mappings)
    final_groups = _group_by_canonical_cluster(clusters, resolved_tag_mappings)
    enriched_clusters: list[CollectiveCluster] = []

    # Phase 2: analyze the same merged evidence set that defines the final output cluster.
    for canonical_key in _ordered_canonical_keys(final_groups):
        grouped_clusters = final_groups[canonical_key]
        merged_diagnoses = sorted(
            [diagnosis for group in grouped_clusters for diagnosis in group.diagnoses],
            key=lambda record: record.task_id,
        )
        merged_cluster = DiagnosisCluster(
            key=canonical_key,
            fault_type=canonical_key[0],
            sub_tags=[canonical_key[1]],
            diagnoses=merged_diagnoses,
            task_ids=[record.task_id for record in merged_diagnoses],
        )
        candidate_clusters = [
            DiagnosisCluster(
                key=other_key,
                fault_type=other_key[0],
                sub_tags=[other_key[1]],
                diagnoses=sorted(
                    [diagnosis for group in final_groups[other_key] for diagnosis in group.diagnoses],
                    key=lambda record: record.task_id,
                ),
                task_ids=sorted(
                    diagnosis.task_id
                    for group in final_groups[other_key]
                    for diagnosis in group.diagnoses
                ),
            )
            for other_key in _ordered_canonical_keys(final_groups)
            if other_key != canonical_key and other_key[0] == canonical_key[0]
        ]
        analysis = CollectiveAnalysisResult.model_validate(
            analyzer(
                _build_collective_payload(
                    cluster=merged_cluster,
                    merged_diagnoses=merged_diagnoses,
                    candidate_clusters=candidate_clusters,
                    tag_mappings=resolved_tag_mappings,
                )
            )
        )
        enriched_clusters.append(
            CollectiveCluster(
                key=canonical_key,
                fault_type=canonical_key[0],
                sub_tags=[canonical_key[1]],
                diagnoses=merged_diagnoses,
                task_ids=[record.task_id for record in merged_diagnoses],
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause=analysis.refined_root_cause,
                    capability_cliff=analysis.capability_cliff,
                    misdiagnosed_ids=analysis.misdiagnosed_ids,
                    misdiagnosis_corrections=analysis.misdiagnosis_corrections,
                    cluster_coherence=analysis.cluster_coherence,
                ),
            )
        )

    return enriched_clusters, resolved_tag_mappings


def default_collective_analyze(payload: dict) -> dict:
    diagnoses = payload["cluster"]["diagnoses"]
    verification_levels = sorted(
        {
            diagnosis["enriched_labels"].get("verification_level")
            for diagnosis in diagnoses
            if diagnosis["enriched_labels"].get("verification_level")
        }
    )
    verification_statuses = sorted(
        {
            diagnosis["enriched_labels"].get("verification_status")
            for diagnosis in diagnoses
            if diagnosis["enriched_labels"].get("verification_status")
        }
    )
    fault_type = payload["cluster"]["fault_type"]
    primary_tag = payload["cluster"]["sub_tags"][0]
    return {
        "refined_root_cause": (
            f"Grouped by {fault_type}/{primary_tag} "
            f"(verification_status={','.join(verification_statuses) or 'unknown'}; "
            f"verification_level={','.join(verification_levels) or 'unknown'})"
        ),
        "capability_cliff": f"{primary_tag} remains unstable under repeated similar tasks.",
        "misdiagnosed_ids": [],
        "misdiagnosis_corrections": {},
        "cluster_coherence": 1.0,
        "semantic_merges": [],
    }


def _initialize_tag_mappings(clusters: list[DiagnosisCluster]) -> dict[str, str]:
    tag_mappings: dict[str, str] = {}
    for cluster in clusters:
        for tag in cluster.sub_tags:
            tag_mappings.setdefault(tag, tag)
    return tag_mappings


def _find_cluster_index(
    clusters: list[DiagnosisCluster],
    fault_type: FaultType,
    primary_tag: str,
) -> int | None:
    for index, cluster in enumerate(clusters):
        cluster_tag = cluster.sub_tags[0] if cluster.sub_tags else "unknown"
        if cluster.fault_type == fault_type and cluster_tag == primary_tag:
            return index
    return None


def _build_collective_payload(
    cluster: DiagnosisCluster,
    merged_diagnoses: list[DiagnosisRecord],
    candidate_clusters: list[DiagnosisCluster],
    tag_mappings: dict[str, str],
) -> dict:
    return {
        "cluster": {
            "fault_type": cluster.fault_type,
            "sub_tags": cluster.sub_tags,
            "task_ids": cluster.task_ids,
            "diagnoses": [diagnosis.model_dump(mode="json") for diagnosis in merged_diagnoses],
        },
        "candidate_clusters": [
            {
                "fault_type": candidate.fault_type,
                "sub_tags": candidate.sub_tags,
                "task_ids": candidate.task_ids,
            }
            for candidate in candidate_clusters
        ],
        "tag_mappings": dict(tag_mappings),
    }


def _resolve_tag_mappings(tag_mappings: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for source_tag in tag_mappings:
        resolved[source_tag] = _resolve_canonical_tag(source_tag, tag_mappings)
    return resolved


def _resolve_canonical_tag(source_tag: str, tag_mappings: dict[str, str]) -> str:
    current = source_tag
    visited: set[str] = set()

    while True:
        target = tag_mappings.get(current, current)
        if target == current or target in visited:
            return target
        visited.add(current)
        current = target


def _cluster_map(clusters: list[DiagnosisCluster]) -> dict[tuple[FaultType, str], DiagnosisCluster]:
    return {cluster.key: cluster for cluster in clusters}


def _ordered_cluster_keys(clusters: list[DiagnosisCluster]) -> list[tuple[FaultType, str]]:
    return [
        cluster.key
        for cluster in sorted(
            clusters,
            key=lambda cluster: (cluster.task_ids[0] if cluster.task_ids else 0, cluster.key),
        )
    ]


def _group_by_canonical_cluster(
    clusters: list[DiagnosisCluster],
    tag_mappings: dict[str, str],
) -> dict[tuple[FaultType, str], list[DiagnosisCluster]]:
    grouped: dict[tuple[FaultType, str], list[DiagnosisCluster]] = {}
    for cluster in clusters:
        primary_tag = cluster.sub_tags[0] if cluster.sub_tags else "unknown"
        canonical_tag = tag_mappings.get(primary_tag, primary_tag)
        grouped.setdefault((cluster.fault_type, canonical_tag), []).append(cluster)
    return grouped


def _ordered_canonical_keys(
    grouped_clusters: dict[tuple[FaultType, str], list[DiagnosisCluster]],
) -> list[tuple[FaultType, str]]:
    return sorted(
        grouped_clusters,
        key=lambda key: min(
            diagnosis.task_id
            for cluster in grouped_clusters[key]
            for diagnosis in cluster.diagnoses
        ),
    )
