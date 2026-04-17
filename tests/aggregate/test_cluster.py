from __future__ import annotations

from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord


def test_cluster_groups_by_fault_type_and_primary_sub_tag() -> None:
    from codemint.aggregate.cluster import cluster_diagnoses

    diagnoses = [
        _diagnosis(1, "implementation", ["off_by_one", "loop_bound"]),
        _diagnosis(2, "implementation", ["off_by_one", "indexing"]),
        _diagnosis(3, "implementation", ["state_leak", "mutation"]),
        _diagnosis(4, "modeling", ["off_by_one", "recurrence"]),
    ]

    clusters = cluster_diagnoses(diagnoses)

    assert [cluster.key for cluster in clusters] == [
        ("implementation", "off_by_one"),
        ("implementation", "state_leak"),
        ("modeling", "off_by_one"),
    ]
    assert clusters[0].task_ids == [1, 2]
    assert clusters[0].sub_tags == ["off_by_one"]
    assert clusters[1].task_ids == [3]
    assert clusters[2].task_ids == [4]


def _diagnosis(task_id: int, fault_type: str, sub_tags: list[str]) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task_id,
        fault_type=fault_type,
        sub_tags=sub_tags,
        severity="medium",
        description=f"Diagnosis {task_id}",
        evidence=DiagnosisEvidence(
            wrong_line="line",
            correct_approach="approach",
            failed_test="test",
        ),
        enriched_labels={},
        confidence=0.8,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )
