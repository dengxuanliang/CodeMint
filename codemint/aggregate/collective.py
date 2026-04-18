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
    tag_mappings = _initialize_tag_mappings(clusters)
    consumed_indices: set[int] = set()
    enriched_clusters: list[CollectiveCluster] = []
    ordered_indices = sorted(
        range(len(clusters)),
        key=lambda index: (clusters[index].task_ids[0] if clusters[index].task_ids else 0, index),
    )

    for index in ordered_indices:
        if index in consumed_indices:
            continue

        cluster = clusters[index]
        analysis = CollectiveAnalysisResult.model_validate(
            analyzer(_build_collective_payload(cluster, clusters, consumed_indices, tag_mappings))
        )
        primary_tag = cluster.sub_tags[0] if cluster.sub_tags else "unknown"
        merged_diagnoses = list(cluster.diagnoses)

        for merge in analysis.semantic_merges:
            if not merge.confirmed or merge.target_tag != primary_tag:
                continue
            source_index = _find_cluster_index(clusters, cluster.fault_type, merge.source_tag)
            if source_index is None or source_index == index or source_index in consumed_indices:
                continue
            consumed_indices.add(source_index)
            merged_cluster = clusters[source_index]
            merged_diagnoses.extend(merged_cluster.diagnoses)
            tag_mappings[merge.source_tag] = merge.target_tag

        sorted_diagnoses = sorted(merged_diagnoses, key=lambda record: record.task_id)
        tag_mappings.setdefault(primary_tag, primary_tag)
        enriched_clusters.append(
            CollectiveCluster(
                key=(cluster.fault_type, primary_tag),
                fault_type=cluster.fault_type,
                sub_tags=[primary_tag],
                diagnoses=sorted_diagnoses,
                task_ids=[record.task_id for record in sorted_diagnoses],
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause=analysis.refined_root_cause,
                    capability_cliff=analysis.capability_cliff,
                    misdiagnosed_ids=analysis.misdiagnosed_ids,
                    misdiagnosis_corrections=analysis.misdiagnosis_corrections,
                    cluster_coherence=analysis.cluster_coherence,
                ),
            )
        )

    return enriched_clusters, tag_mappings


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
    all_clusters: list[DiagnosisCluster],
    consumed_indices: set[int],
    tag_mappings: dict[str, str],
) -> dict:
    return {
        "cluster": {
            "fault_type": cluster.fault_type,
            "sub_tags": cluster.sub_tags,
            "task_ids": cluster.task_ids,
            "diagnoses": [diagnosis.model_dump(mode="json") for diagnosis in cluster.diagnoses],
        },
        "candidate_clusters": [
            {
                "fault_type": candidate.fault_type,
                "sub_tags": candidate.sub_tags,
                "task_ids": candidate.task_ids,
            }
            for index, candidate in enumerate(all_clusters)
            if candidate.key != cluster.key and index not in consumed_indices and candidate.fault_type == cluster.fault_type
        ],
        "tag_mappings": dict(tag_mappings),
    }
