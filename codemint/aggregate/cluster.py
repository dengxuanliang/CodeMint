from __future__ import annotations

from dataclasses import dataclass

from codemint.models.diagnosis import DiagnosisRecord, FaultType


@dataclass(frozen=True, slots=True)
class DiagnosisCluster:
    key: tuple[FaultType, str]
    fault_type: FaultType
    sub_tags: list[str]
    diagnoses: list[DiagnosisRecord]
    task_ids: list[int]


def cluster_diagnoses(diagnoses: list[DiagnosisRecord]) -> list[DiagnosisCluster]:
    grouped: dict[tuple[FaultType, str], list[DiagnosisRecord]] = {}
    for diagnosis in diagnoses:
        primary_sub_tag = diagnosis.sub_tags[0] if diagnosis.sub_tags else "unknown"
        key = (diagnosis.fault_type, primary_sub_tag)
        grouped.setdefault(key, []).append(diagnosis)

    clusters: list[DiagnosisCluster] = []
    for key in sorted(grouped):
        grouped_diagnoses = sorted(grouped[key], key=lambda record: record.task_id)
        clusters.append(
            DiagnosisCluster(
                key=key,
                fault_type=key[0],
                sub_tags=[key[1]],
                diagnoses=grouped_diagnoses,
                task_ids=[record.task_id for record in grouped_diagnoses],
            )
        )
    return clusters
