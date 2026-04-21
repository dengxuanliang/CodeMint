from __future__ import annotations

import json
from pathlib import Path

from codemint.config import CodeMintConfig
from codemint.diagnose.clustering import DiagnoseCluster, cluster_fingerprints
from codemint.diagnose.confirm import ConfirmAnalyzer
from codemint.diagnose.deep import DeepAnalyzer
from codemint.diagnose.fingerprint import build_fingerprint
from codemint.diagnose.item_mode import (
    _default_confirm_analyzer,
    _default_deep_analyzer,
    _load_existing_diagnoses,
    _validate_unique_task_ids,
    diagnose_single_task,
)
from codemint.diagnose.propagation import PropagationResult, propagate_cluster_diagnoses
from codemint.diagnose.resume import find_missing_task_ids
from codemint.io.jsonl import append_jsonl
from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.builtin import DiagnosisRule
from codemint.rules.custom import build_rules


def run_clustered_mode(
    tasks: list[TaskRecord],
    output_path: Path,
    rules: list[DiagnosisRule] | None = None,
    *,
    config: CodeMintConfig | None = None,
    confirm_analyzer: ConfirmAnalyzer | None = None,
    deep_analyzer: DeepAnalyzer | None = None,
) -> list[DiagnosisRecord]:
    active_rules = build_rules() if rules is None else rules
    resolved_config = config or CodeMintConfig()
    diagnose_config = resolved_config.diagnose
    confirmer = confirm_analyzer or _default_confirm_analyzer(resolved_config)
    deep = deep_analyzer or _default_deep_analyzer(resolved_config)
    _validate_unique_task_ids(tasks)

    existing_diagnoses = _load_existing_diagnoses(output_path)
    missing_task_ids = set(find_missing_task_ids(output_path, [task.task_id for task in tasks]))
    missing_tasks = [task for task in tasks if task.task_id in missing_task_ids]
    if not missing_tasks:
        return existing_diagnoses

    tasks_by_id = {task.task_id: task for task in missing_tasks}
    fingerprints = [build_fingerprint(task) for task in missing_tasks]
    clusters = cluster_fingerprints(
        fingerprints,
        threshold=diagnose_config.clustering_threshold,
        representatives=diagnose_config.cluster_representatives,
    )

    new_diagnoses: list[DiagnosisRecord] = []
    cluster_payloads: list[dict] = []
    for cluster in clusters:
        representative_diagnoses = [
            diagnose_single_task(
                tasks_by_id[task_id],
                rules=active_rules,
                confirm_analyzer=confirmer,
                deep_analyzer=deep,
            )
            for task_id in cluster.representative_task_ids
        ]
        propagation = propagate_cluster_diagnoses(
            cluster,
            representative_diagnoses=representative_diagnoses,
            low_confidence_threshold=diagnose_config.low_confidence_threshold,
            rediagnose_low_confidence=diagnose_config.rediagnose_low_confidence,
            max_cluster_size_for_propagation=diagnose_config.max_cluster_size_for_propagation,
        )
        fallback_diagnoses = [
            diagnose_single_task(
                tasks_by_id[task_id],
                rules=active_rules,
                confirm_analyzer=confirmer,
                deep_analyzer=deep,
            )
            for task_id in propagation.fallback_task_ids
        ]
        new_diagnoses.extend(representative_diagnoses)
        new_diagnoses.extend(propagation.propagated)
        new_diagnoses.extend(fallback_diagnoses)
        cluster_payloads.append(_cluster_payload(cluster, propagation))

    ordered_new = sorted(new_diagnoses, key=lambda diagnosis: diagnosis.task_id)
    if ordered_new:
        append_jsonl(output_path, [diagnosis.model_dump(mode="json") for diagnosis in ordered_new])
    _write_cluster_artifact(output_path.parent / "diagnose_clusters.json", clusters, cluster_payloads)
    return existing_diagnoses + ordered_new


def _cluster_payload(cluster: DiagnoseCluster, propagation: PropagationResult) -> dict:
    return {
        "cluster_id": cluster.cluster_id,
        "member_task_ids": cluster.member_task_ids,
        "representative_task_ids": cluster.representative_task_ids,
        "cluster_size": len(cluster.member_task_ids),
        "min_similarity": cluster.min_similarity,
        "fingerprint_summary": cluster.fingerprint_summary,
        "fallback_task_ids": propagation.fallback_task_ids,
        "propagated_task_ids": [diagnosis.task_id for diagnosis in propagation.propagated],
    }


def _write_cluster_artifact(
    path: Path,
    clusters: list[DiagnoseCluster],
    cluster_payloads: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "cluster_count": len(clusters),
            "task_count": sum(len(cluster.member_task_ids) for cluster in clusters),
        },
        "clusters": cluster_payloads,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
