from __future__ import annotations

from pathlib import Path

from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord


def test_semantic_tag_merge_records_mapping_when_model_confirms() -> None:
    from codemint.aggregate.collective import apply_collective_diagnosis
    from codemint.aggregate.cluster import cluster_diagnoses

    diagnoses = [
        _diagnosis(1, "implementation", ["off_by_one"]),
        _diagnosis(2, "implementation", ["index_bounds"]),
    ]

    def collective_stub(payload: dict) -> dict:
        assert payload["cluster"]["task_ids"] == [1]
        return {
            "refined_root_cause": "Boundary reasoning is inconsistent.",
            "capability_cliff": "State-index alignment breaks near array edges.",
            "misdiagnosed_ids": [],
            "misdiagnosis_corrections": {},
            "cluster_coherence": 0.92,
            "semantic_merges": [
                {
                    "source_tag": "index_bounds",
                    "target_tag": "off_by_one",
                    "confirmed": True,
                }
            ],
        }

    enriched_clusters, tag_mappings = apply_collective_diagnosis(
        cluster_diagnoses(diagnoses),
        collective_stub,
    )

    assert [cluster.task_ids for cluster in enriched_clusters] == [[1, 2]]
    assert enriched_clusters[0].sub_tags == ["off_by_one"]
    assert tag_mappings == {"index_bounds": "off_by_one", "off_by_one": "off_by_one"}


def test_collective_diagnosis_reclassifies_misdiagnosed_ids(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(10, "implementation", ["loop_bound"]),
        _diagnosis(11, "implementation", ["loop_bound"]),
    ]
    output_path = tmp_path / "weaknesses.json"

    def collective_stub(payload: dict) -> dict:
        assert payload["cluster"]["task_ids"] == [10, 11]
        return {
            "refined_root_cause": "Loop invariants are not maintained.",
            "capability_cliff": "Longer iterations break accumulator reasoning.",
            "misdiagnosed_ids": [11],
            "misdiagnosis_corrections": {"11": "modeling:state_tracking"},
            "cluster_coherence": 0.61,
            "semantic_merges": [],
        }

    report = run_aggregate(
        diagnoses,
        output_path,
        collective_analyze=collective_stub,
    )

    assert len(report.weaknesses) == 2
    implementation = next(entry for entry in report.weaknesses if entry.fault_type == "implementation")
    modeling = next(entry for entry in report.weaknesses if entry.fault_type == "modeling")
    assert implementation.frequency == 1
    assert implementation.sample_task_ids == [10]
    assert implementation.collective_diagnosis.misdiagnosed_ids == [11]
    assert implementation.collective_diagnosis.misdiagnosis_corrections == {"11": "modeling:state_tracking"}
    assert modeling.sub_tags == ["state_tracking"]
    assert modeling.sample_task_ids == [11]


def test_semantic_tag_merge_chains_normalize_to_final_canonical_tag() -> None:
    from codemint.aggregate.collective import apply_collective_diagnosis
    from codemint.aggregate.cluster import cluster_diagnoses

    diagnoses = [
        _diagnosis(1, "implementation", ["tag_c"]),
        _diagnosis(2, "implementation", ["tag_b"]),
        _diagnosis(3, "implementation", ["tag_a"]),
    ]

    def collective_stub(payload: dict) -> dict:
        task_ids = payload["cluster"]["task_ids"]
        if task_ids == [1]:
            return {
                "refined_root_cause": "Canonical root cause.",
                "capability_cliff": "Canonical cliff.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.95,
                "semantic_merges": [
                    {"source_tag": "tag_b", "target_tag": "tag_c", "confirmed": True},
                ],
            }
        if task_ids == [3]:
            return {
                "refined_root_cause": "Alias root cause.",
                "capability_cliff": "Alias cliff.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.88,
                "semantic_merges": [
                    {"source_tag": "tag_a", "target_tag": "tag_b", "confirmed": True},
                ],
            }
        raise AssertionError(f"unexpected payload order: {task_ids}")

    enriched_clusters, tag_mappings = apply_collective_diagnosis(
        cluster_diagnoses(diagnoses),
        collective_stub,
    )

    assert [cluster.task_ids for cluster in enriched_clusters] == [[1, 2], [3]]
    assert tag_mappings == {"tag_a": "tag_c", "tag_b": "tag_c", "tag_c": "tag_c"}


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
