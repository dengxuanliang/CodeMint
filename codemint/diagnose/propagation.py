from __future__ import annotations

from dataclasses import dataclass

from codemint.diagnose.clustering import DiagnoseCluster
from codemint.models.diagnosis import DiagnosisRecord


@dataclass(frozen=True, slots=True)
class PropagationResult:
    propagated: list[DiagnosisRecord]
    fallback_task_ids: list[int]


def propagate_cluster_diagnoses(
    cluster: DiagnoseCluster,
    *,
    representative_diagnoses: list[DiagnosisRecord],
    low_confidence_threshold: float,
    rediagnose_low_confidence: bool,
    max_cluster_size_for_propagation: int,
) -> PropagationResult:
    member_task_ids = sorted(cluster.member_task_ids)
    representative_ids = set(cluster.representative_task_ids)
    propagation_targets = [task_id for task_id in member_task_ids if task_id not in representative_ids]

    if not propagation_targets:
        return PropagationResult(propagated=[], fallback_task_ids=[])
    if len(member_task_ids) > max_cluster_size_for_propagation:
        return PropagationResult(propagated=[], fallback_task_ids=propagation_targets)
    if not _representatives_agree(representative_diagnoses):
        return PropagationResult(propagated=[], fallback_task_ids=propagation_targets)

    representative = sorted(representative_diagnoses, key=lambda item: item.task_id)[0]
    if rediagnose_low_confidence and representative.confidence < low_confidence_threshold:
        return PropagationResult(propagated=[], fallback_task_ids=propagation_targets)

    propagated = [
        _copy_for_member(representative, cluster.cluster_id, task_id)
        for task_id in propagation_targets
    ]
    return PropagationResult(propagated=propagated, fallback_task_ids=[])


def _representatives_agree(diagnoses: list[DiagnosisRecord]) -> bool:
    if not diagnoses:
        return False
    primary_tags = {
        diagnosis.sub_tags[0] if diagnosis.sub_tags else "unknown"
        for diagnosis in diagnoses
    }
    fault_types = {diagnosis.fault_type for diagnosis in diagnoses}
    return len(primary_tags) == 1 and len(fault_types) == 1


def _copy_for_member(
    representative: DiagnosisRecord,
    cluster_id: str,
    task_id: int,
) -> DiagnosisRecord:
    labels = {
        **representative.enriched_labels,
        "processing_mode": "clustered",
        "diagnosis_origin": "propagated",
        "cluster_id": cluster_id,
        "representative_task_id": str(representative.task_id),
    }
    return representative.model_copy(
        deep=True,
        update={
            "task_id": task_id,
            "enriched_labels": labels,
        },
    )
