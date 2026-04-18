from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from codemint.aggregate.cluster import cluster_diagnoses
from codemint.aggregate.repair import (
    VerificationLevel,
    VerificationResult,
    default_verify,
    repair_diagnosis,
)
from codemint.models.diagnosis import DiagnosisRecord
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
    report = _build_report(clusters, repaired)
    _write_report(output_path, report)
    return report


def _build_report(clusters, repaired: list[DiagnosisRecord]) -> WeaknessReport:
    weaknesses: list[WeaknessEntry] = []
    verification_levels = sorted(
        {
            diagnosis.enriched_labels.get("verification_level")
            for diagnosis in repaired
            if diagnosis.enriched_labels.get("verification_level")
        }
    )
    verification_statuses = sorted(
        {
            diagnosis.enriched_labels.get("verification_status")
            for diagnosis in repaired
            if diagnosis.enriched_labels.get("verification_status")
        }
    )
    for index, cluster in enumerate(clusters, start=1):
        weaknesses.append(
            WeaknessEntry(
                rank=index,
                fault_type=cluster.fault_type,
                sub_tags=cluster.sub_tags,
                frequency=len(cluster.diagnoses),
                sample_task_ids=cluster.task_ids[:3],
                trainability=1.0,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause=(
                        f"Grouped by {cluster.fault_type}/{cluster.sub_tags[0]} "
                        f"(verification_status={','.join(verification_statuses) or 'unknown'}; "
                        f"verification_level={','.join(verification_levels) or 'unknown'})"
                    ),
                    capability_cliff="pending_task_9",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=1.0,
                ),
            )
        )

    ranks = [entry.rank for entry in weaknesses]
    tag_mappings = {
        entry.sub_tags[0]: entry.sub_tags[0]
        for entry in weaknesses
        if entry.sub_tags
    }
    return WeaknessReport(
        weaknesses=weaknesses,
        rankings=RankingSet(
            by_frequency=ranks,
            by_difficulty=ranks,
            by_trainability=ranks,
        ),
        causal_chains=[
            CausalChain(
                root="pending_task_9",
                downstream=[entry.sub_tags[0] for entry in weaknesses if entry.sub_tags],
                training_priority="pending_task_9",
            )
        ]
        if weaknesses
        else [],
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
