from __future__ import annotations

from codemint.diagnose.clustering import DiagnoseCluster
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord


def test_propagation_copies_consistent_representative_diagnosis_to_members() -> None:
    from codemint.diagnose.propagation import propagate_cluster_diagnoses

    cluster = _cluster([1, 2, 3], representatives=[1])
    representative = _diagnosis(1, sub_tags=["syntax_error"], confidence=0.9)

    result = propagate_cluster_diagnoses(
        cluster,
        representative_diagnoses=[representative],
        low_confidence_threshold=0.55,
        rediagnose_low_confidence=True,
        max_cluster_size_for_propagation=50,
    )

    assert result.fallback_task_ids == []
    assert [diagnosis.task_id for diagnosis in result.propagated] == [2, 3]
    assert result.propagated[0].sub_tags == ["syntax_error"]
    assert result.propagated[0].enriched_labels["diagnosis_origin"] == "propagated"
    assert result.propagated[0].enriched_labels["cluster_id"] == "cluster-0001"


def test_propagation_falls_back_when_representatives_disagree() -> None:
    from codemint.diagnose.propagation import propagate_cluster_diagnoses

    cluster = _cluster([1, 2, 3], representatives=[1, 2])
    representatives = [
        _diagnosis(1, sub_tags=["syntax_error"], confidence=0.9),
        _diagnosis(2, sub_tags=["markdown_formatting"], confidence=0.9),
    ]

    result = propagate_cluster_diagnoses(
        cluster,
        representative_diagnoses=representatives,
        low_confidence_threshold=0.55,
        rediagnose_low_confidence=True,
        max_cluster_size_for_propagation=50,
    )

    assert result.propagated == []
    assert result.fallback_task_ids == [3]


def test_propagation_falls_back_for_low_confidence_when_enabled() -> None:
    from codemint.diagnose.propagation import propagate_cluster_diagnoses

    cluster = _cluster([1, 2], representatives=[1])
    representative = _diagnosis(1, sub_tags=["missing_code_block"], confidence=0.4)

    result = propagate_cluster_diagnoses(
        cluster,
        representative_diagnoses=[representative],
        low_confidence_threshold=0.55,
        rediagnose_low_confidence=True,
        max_cluster_size_for_propagation=50,
    )

    assert result.propagated == []
    assert result.fallback_task_ids == [2]


def _cluster(
    member_task_ids: list[int],
    *,
    representatives: list[int],
) -> DiagnoseCluster:
    return DiagnoseCluster(
        cluster_id="cluster-0001",
        member_task_ids=member_task_ids,
        representative_task_ids=representatives,
        fingerprint_summary={"entry_point_hint": "solve"},
        min_similarity=1.0,
    )


def _diagnosis(
    task_id: int,
    *,
    sub_tags: list[str],
    confidence: float,
) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task_id,
        fault_type="implementation",
        sub_tags=sub_tags,
        severity="medium",
        description="Representative diagnosis.",
        evidence=DiagnosisEvidence(
            wrong_line="def solve(x)",
            correct_approach="Return valid executable code.",
            failed_test="assert solve(1) == 2",
        ),
        enriched_labels={},
        confidence=confidence,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )
