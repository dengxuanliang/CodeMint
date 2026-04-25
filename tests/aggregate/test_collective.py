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
    seen_payloads: list[list[int]] = []

    def collective_stub(payload: dict) -> dict:
        seen_payloads.append(payload["cluster"]["task_ids"])
        if payload["cluster"]["task_ids"] == [1]:
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
        if payload["cluster"]["task_ids"] == [1, 2]:
            return {
                "refined_root_cause": "Merged boundary reasoning issue.",
                "capability_cliff": "Merged boundary cliff.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.95,
                "semantic_merges": [],
            }
        raise AssertionError(f"unexpected payload: {payload['cluster']['task_ids']}")

    enriched_clusters, tag_mappings = apply_collective_diagnosis(
        cluster_diagnoses(diagnoses),
        collective_stub,
    )

    assert [cluster.task_ids for cluster in enriched_clusters] == [[1, 2]]
    assert enriched_clusters[0].sub_tags == ["off_by_one"]
    assert enriched_clusters[0].collective_diagnosis.refined_root_cause == "Merged boundary reasoning issue."
    assert tag_mappings == {"index_bounds": "off_by_one", "off_by_one": "off_by_one"}
    assert seen_payloads == [[1], [1, 2]]


def test_collective_diagnosis_records_misdiagnosed_ids_without_reclassification(tmp_path: Path) -> None:
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

    assert len(report.weaknesses) == 1
    implementation = report.weaknesses[0]
    assert implementation.fault_type == "implementation"
    assert implementation.sub_tags == ["logic_error"]
    assert implementation.frequency == 2
    assert implementation.sample_task_ids == [10, 11]
    assert implementation.collective_diagnosis.misdiagnosed_ids == [11]
    assert implementation.collective_diagnosis.misdiagnosis_corrections == {"11": "modeling:state_tracking"}


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
        if task_ids == [1, 2, 3]:
            return {
                "refined_root_cause": "Final canonical root cause.",
                "capability_cliff": "Final canonical cliff.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.98,
                "semantic_merges": [],
            }
        raise AssertionError(f"unexpected payload order: {task_ids}")

    enriched_clusters, tag_mappings = apply_collective_diagnosis(
        cluster_diagnoses(diagnoses),
        collective_stub,
    )

    assert [cluster.task_ids for cluster in enriched_clusters] == [[1, 2, 3]]
    assert enriched_clusters[0].collective_diagnosis.refined_root_cause == "Final canonical root cause."
    assert tag_mappings == {"tag_a": "tag_c", "tag_b": "tag_c", "tag_c": "tag_c"}


def test_emitted_cluster_cannot_survive_later_canonical_merge(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(1, "implementation", ["tag_c"]),
        _diagnosis(2, "implementation", ["tag_b"]),
        _diagnosis(3, "implementation", ["tag_a"]),
    ]
    output_path = tmp_path / "weaknesses.json"

    def collective_stub(payload: dict) -> dict:
        task_ids = payload["cluster"]["task_ids"]
        if task_ids == [1]:
            return {
                "refined_root_cause": "Final canonical cause.",
                "capability_cliff": "Final canonical cliff.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.96,
                "semantic_merges": [
                    {"source_tag": "tag_b", "target_tag": "tag_c", "confirmed": True},
                ],
            }
        if task_ids == [3]:
            return {
                "refined_root_cause": "Intermediate alias cause.",
                "capability_cliff": "Intermediate alias cliff.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.84,
                "semantic_merges": [
                    {"source_tag": "tag_a", "target_tag": "tag_b", "confirmed": True},
                ],
            }
        if task_ids == [1, 2, 3]:
            return {
                "refined_root_cause": "Final canonical cause.",
                "capability_cliff": "Final canonical cliff.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.99,
                "semantic_merges": [],
            }
        raise AssertionError(f"unexpected payload order: {task_ids}")

    report = run_aggregate(
        diagnoses,
        output_path,
        collective_analyze=collective_stub,
    )

    assert len(report.weaknesses) == 1
    assert report.weaknesses[0].sub_tags == ["logic_error"]
    assert report.weaknesses[0].sample_task_ids == [1, 2, 3]
    assert report.tag_mappings == {"tag_a": "logic_error", "tag_b": "logic_error", "tag_c": "logic_error"}


def test_collective_diagnosis_reflects_merged_evidence_and_task_ids(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(10, "implementation", ["off_by_one"]),
        _diagnosis(11, "implementation", ["index_bounds"]),
    ]
    output_path = tmp_path / "weaknesses.json"
    seen_payloads: list[list[int]] = []

    def collective_stub(payload: dict) -> dict:
        seen_payloads.append(payload["cluster"]["task_ids"])
        if payload["cluster"]["task_ids"] == [10]:
            return {
                "refined_root_cause": "Pre-merge root cause.",
                "capability_cliff": "Pre-merge cliff.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.91,
                "semantic_merges": [
                    {"source_tag": "index_bounds", "target_tag": "off_by_one", "confirmed": True},
                ],
            }
        if payload["cluster"]["task_ids"] == [10, 11]:
            return {
                "refined_root_cause": "Merged root cause for tasks 10 and 11.",
                "capability_cliff": "Merged cliff for tasks 10 and 11.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.97,
                "semantic_merges": [],
            }
        raise AssertionError(f"unexpected collective payload: {payload['cluster']['task_ids']}")

    report = run_aggregate(
        diagnoses,
        output_path,
        collective_analyze=collective_stub,
    )

    assert seen_payloads == [[10], [10, 11]]
    assert len(report.weaknesses) == 1
    assert report.weaknesses[0].frequency == 2
    assert report.weaknesses[0].sample_task_ids == [10, 11]
    assert report.weaknesses[0].collective_diagnosis.refined_root_cause == "Merged root cause for tasks 10 and 11."
    assert report.weaknesses[0].collective_diagnosis.capability_cliff == "Merged cliff for tasks 10 and 11."


def test_malformed_misdiagnosis_corrections_are_ignored_safely(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(21, "implementation", ["loop_bound"]),
        _diagnosis(22, "implementation", ["loop_bound"]),
    ]
    output_path = tmp_path / "weaknesses.json"

    def collective_stub(payload: dict) -> dict:
        assert payload["cluster"]["task_ids"] == [21, 22]
        return {
            "refined_root_cause": "Loop boundary issue.",
            "capability_cliff": "Loop boundary cliff.",
            "misdiagnosed_ids": [22],
            "misdiagnosis_corrections": {
                "abc": "modeling:state_tracking",
                "22": "not_a_valid_correction",
                "21": "invalid_fault:tag_x",
            },
            "cluster_coherence": 0.7,
            "semantic_merges": [],
        }

    report = run_aggregate(
        diagnoses,
        output_path,
        collective_analyze=collective_stub,
    )

    assert len(report.weaknesses) == 1
    assert report.weaknesses[0].fault_type == "implementation"
    assert report.weaknesses[0].sub_tags == ["logic_error"]
    assert report.weaknesses[0].sample_task_ids == [21, 22]
    assert report.weaknesses[0].collective_diagnosis.misdiagnosis_corrections == {
        "abc": "modeling:state_tracking",
        "22": "not_a_valid_correction",
        "21": "invalid_fault:tag_x",
    }


def test_collective_corrections_do_not_reclassify_cluster_members(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(30, "implementation", ["off_by_one"]),
        _diagnosis(31, "implementation", ["loop_bound"]),
    ]
    output_path = tmp_path / "weaknesses.json"

    def collective_stub(payload: dict) -> dict:
        task_ids = payload["cluster"]["task_ids"]
        if task_ids == [30]:
            return {
                "refined_root_cause": "Boundary reasoning issue.",
                "capability_cliff": "Boundary reasoning cliff.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.93,
                "semantic_merges": [
                    {"source_tag": "index_bounds", "target_tag": "off_by_one", "confirmed": True},
                ],
            }
        if task_ids == [31]:
            return {
                "refined_root_cause": "Loop issue.",
                "capability_cliff": "Loop cliff.",
                "misdiagnosed_ids": [31],
                "misdiagnosis_corrections": {"31": "modeling:index_bounds"},
                "cluster_coherence": 0.7,
                "semantic_merges": [],
            }
        if task_ids == [30, 31]:
            return {
                "refined_root_cause": "Merged canonical issue.",
                "capability_cliff": "Merged canonical cliff.",
                "misdiagnosed_ids": [],
                "misdiagnosis_corrections": {},
                "cluster_coherence": 0.97,
                "semantic_merges": [],
            }
        raise AssertionError(f"unexpected payload order: {task_ids}")

    report = run_aggregate(
        diagnoses,
        output_path,
        collective_analyze=collective_stub,
    )

    implementation = next(entry for entry in report.weaknesses if entry.fault_type == "implementation")
    assert implementation.sub_tags == ["logic_error"]
    assert implementation.sample_task_ids == [30, 31]
    assert report.tag_mappings["index_bounds"] == "logic_error"


def test_collective_diagnosis_does_not_merge_function_name_mismatch() -> None:
    from codemint.aggregate.collective import apply_collective_diagnosis
    from codemint.aggregate.cluster import cluster_diagnoses

    diagnoses = [
        _diagnosis(41, "surface", ["function_name_mismatch"]),
        _diagnosis(42, "surface", ["wrong_function_name"]),
    ]

    def collective_stub(payload: dict) -> dict:
        return {
            "refined_root_cause": "Interface contract issue.",
            "capability_cliff": "Public entry point becomes unstable.",
            "misdiagnosed_ids": [],
            "misdiagnosis_corrections": {},
            "cluster_coherence": 0.85,
            "semantic_merges": [
                {
                    "source_tag": "wrong_function_name",
                    "target_tag": "function_name_mismatch",
                    "confirmed": True,
                }
            ],
        }

    enriched_clusters, tag_mappings = apply_collective_diagnosis(
        cluster_diagnoses(diagnoses),
        collective_stub,
    )

    assert [cluster.sub_tags for cluster in enriched_clusters] == [["function_name_mismatch"], ["wrong_function_name"]]
    assert tag_mappings["function_name_mismatch"] == "function_name_mismatch"
    assert tag_mappings["wrong_function_name"] == "wrong_function_name"


def test_collective_diagnosis_does_not_merge_markdown_formatting() -> None:
    from codemint.aggregate.collective import apply_collective_diagnosis
    from codemint.aggregate.cluster import cluster_diagnoses

    diagnoses = [
        _diagnosis(51, "surface", ["markdown_formatting"]),
        _diagnosis(52, "surface", ["extraneous_characters"]),
    ]

    def collective_stub(payload: dict) -> dict:
        return {
            "refined_root_cause": "Formatting issue.",
            "capability_cliff": "Raw-output formatting drifts.",
            "misdiagnosed_ids": [],
            "misdiagnosis_corrections": {},
            "cluster_coherence": 0.82,
            "semantic_merges": [
                {
                    "source_tag": "extraneous_characters",
                    "target_tag": "markdown_formatting",
                    "confirmed": True,
                }
            ],
        }

    enriched_clusters, tag_mappings = apply_collective_diagnosis(
        cluster_diagnoses(diagnoses),
        collective_stub,
    )

    assert [cluster.sub_tags for cluster in enriched_clusters] == [["markdown_formatting"], ["extraneous_characters"]]
    assert tag_mappings["markdown_formatting"] == "markdown_formatting"
    assert tag_mappings["extraneous_characters"] == "extraneous_characters"


def test_collective_diagnosis_does_not_merge_missing_code_block_into_syntax_error() -> None:
    from codemint.aggregate.collective import apply_collective_diagnosis
    from codemint.aggregate.cluster import cluster_diagnoses

    diagnoses = [
        _diagnosis(61, "implementation", ["missing_code_block"]),
        _diagnosis(62, "implementation", ["syntax_error"]),
    ]

    def collective_stub(payload: dict) -> dict:
        return {
            "refined_root_cause": "Non-executable response issue.",
            "capability_cliff": "Code output fails to execute.",
            "misdiagnosed_ids": [],
            "misdiagnosis_corrections": {},
            "cluster_coherence": 0.8,
            "semantic_merges": [
                {
                    "source_tag": "missing_code_block",
                    "target_tag": "syntax_error",
                    "confirmed": True,
                }
            ],
        }

    enriched_clusters, tag_mappings = apply_collective_diagnosis(
        cluster_diagnoses(diagnoses),
        collective_stub,
    )

    assert [cluster.sub_tags for cluster in enriched_clusters] == [["missing_code_block"], ["syntax_error"]]
    assert tag_mappings["missing_code_block"] == "missing_code_block"
    assert tag_mappings["syntax_error"] == "syntax_error"


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
