import pytest

from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.run_metadata import RunMetadata
from codemint.models.spec import SpecRecord
from codemint.models.task import TaskRecord
from codemint.models.weakness import WeaknessReport


def test_task_record_dataclass_preserves_fields() -> None:
    record = TaskRecord(
        task_id=101,
        content="Solve the problem.",
        canonical_solution="def solve(): pass",
        completion="def solve(): return 1",
        test_code="assert solve() == 2",
        labels={"difficulty": "hard", "category": "dp"},
        accepted=False,
        metrics={"latency_ms": 12.5},
        extra={"source": "eval_log.jsonl"},
    )

    assert record.task_id == 101
    assert record.labels["difficulty"] == "hard"
    assert record.extra["source"] == "eval_log.jsonl"


def test_task_record_requires_spec_fields() -> None:
    try:
        TaskRecord(
            task_id=101,
            content="Solve the problem.",
            canonical_solution="def solve(): pass",
            completion="def solve(): return 1",
            test_code="assert solve() == 2",
        )
    except TypeError as exc:
        message = str(exc)
    else:
        raise AssertionError("TaskRecord accepted missing required spec fields")

    assert "labels" in message
    assert "accepted" in message
    assert "metrics" in message
    assert "extra" in message


def test_diagnosis_round_trip() -> None:
    diagnosis = DiagnosisRecord.model_validate(
        {
            "task_id": 101,
            "fault_type": "modeling",
            "sub_tags": ["chose_greedy_over_dp"],
            "severity": "high",
            "description": "Greedy was used where DP was required.",
            "evidence": {
                "wrong_line": "result = greedy(items)",
                "correct_approach": "Use dynamic programming.",
                "failed_test": "expected 120, got 100",
            },
            "enriched_labels": {
                "algorithm_type": "dynamic_programming",
                "pattern": "0-1_knapsack_variant",
            },
            "confidence": 0.85,
            "diagnosis_source": "model_deep",
            "prompt_version": "v2",
        }
    )

    assert DiagnosisRecord.model_validate_json(diagnosis.model_dump_json()) == diagnosis


def test_diagnosis_accepts_non_failure_source_and_property() -> None:
    diagnosis = DiagnosisRecord.model_validate(
        {
            "task_id": 202,
            "fault_type": "implementation",
            "sub_tags": ["correct_output"],
            "severity": "low",
            "description": "The solution is correct.",
            "evidence": {
                "wrong_line": "N/A",
                "correct_approach": "Already correct.",
                "failed_test": "N/A",
            },
            "enriched_labels": {
                "status": "correct_solution",
                "test_result": "pass",
            },
            "confidence": 0.99,
            "diagnosis_source": "non_failure",
            "prompt_version": "v2",
        }
    )

    assert diagnosis.diagnosis_source == "non_failure"
    assert diagnosis.is_failure is False


def test_diagnosis_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        DiagnosisRecord.model_validate(
            {
                "task_id": 101,
                "fault_type": "modeling",
                "sub_tags": ["chose_greedy_over_dp"],
                "severity": "high",
                "description": "Greedy was used where DP was required.",
                "evidence": {
                    "wrong_line": "result = greedy(items)",
                    "correct_approach": "Use dynamic programming.",
                    "failed_test": "expected 120, got 100",
                    "unexpected": "extra",
                },
                "enriched_labels": {
                    "algorithm_type": "dynamic_programming",
                    "pattern": "0-1_knapsack_variant",
                },
                "confidence": 0.85,
                "diagnosis_source": "model_deep",
                "prompt_version": "v2",
            }
        )


def test_diagnosis_rejects_out_of_bounds_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        DiagnosisRecord.model_validate(
            {
                "task_id": 101,
                "fault_type": "modeling",
                "sub_tags": ["chose_greedy_over_dp"],
                "severity": "high",
                "description": "Greedy was used where DP was required.",
                "evidence": {
                    "wrong_line": "result = greedy(items)",
                    "correct_approach": "Use dynamic programming.",
                    "failed_test": "expected 120, got 100",
                },
                "enriched_labels": {
                    "algorithm_type": "dynamic_programming",
                    "pattern": "0-1_knapsack_variant",
                },
                "confidence": 1.5,
                "diagnosis_source": "model_deep",
                "prompt_version": "v2",
            }
        )


def test_weakness_report_round_trip() -> None:
    report = WeaknessReport.model_validate(
        {
            "weaknesses": [
                {
                    "rank": 1,
                    "fault_type": "modeling",
                    "sub_tags": ["chose_greedy_over_dp"],
                    "frequency": 45,
                    "sample_task_ids": [101, 203, 307],
                    "trainability": 0.9,
                    "collective_diagnosis": {
                        "refined_root_cause": "Cannot identify overlapping subproblems.",
                        "capability_cliff": "Fails when n>20.",
                        "misdiagnosed_ids": [405],
                        "misdiagnosis_corrections": {"405": "Actually comprehension error."},
                        "cluster_coherence": 0.88,
                    },
                }
            ],
            "rankings": {
                "by_frequency": [1, 3, 2],
                "by_difficulty": [2, 1, 3],
                "by_trainability": [1, 2, 3],
            },
            "causal_chains": [
                {
                    "root": "fails_to_identify_overlapping_subproblems",
                    "downstream": ["chose_greedy_over_dp", "missed_memoization"],
                    "training_priority": "Train root cause first.",
                }
            ],
            "tag_mappings": {
                "greedy_instead_of_dp": "chose_greedy_over_dp",
                "wrong_greedy": "chose_greedy_over_dp",
            },
        }
    )

    assert WeaknessReport.model_validate_json(report.model_dump_json()) == report


def test_weakness_report_rejects_out_of_bounds_scores() -> None:
    with pytest.raises(ValueError, match="trainability|cluster_coherence"):
        WeaknessReport.model_validate(
            {
                "weaknesses": [
                    {
                        "rank": 1,
                        "fault_type": "modeling",
                        "sub_tags": ["chose_greedy_over_dp"],
                        "frequency": 45,
                        "sample_task_ids": [101, 203, 307],
                        "trainability": -0.1,
                        "collective_diagnosis": {
                            "refined_root_cause": "Cannot identify overlapping subproblems.",
                            "capability_cliff": "Fails when n>20.",
                            "misdiagnosed_ids": [405],
                            "misdiagnosis_corrections": {"405": "Actually comprehension error."},
                            "cluster_coherence": 1.2,
                        },
                    }
                ],
                "rankings": {
                    "by_frequency": [1, 3, 2],
                    "by_difficulty": [2, 1, 3],
                    "by_trainability": [1, 2, 3],
                },
                "causal_chains": [
                    {
                        "root": "fails_to_identify_overlapping_subproblems",
                        "downstream": ["chose_greedy_over_dp", "missed_memoization"],
                        "training_priority": "Train root cause first.",
                    }
                ],
                "tag_mappings": {
                    "greedy_instead_of_dp": "chose_greedy_over_dp",
                },
            }
        )


def test_spec_round_trip() -> None:
    spec = SpecRecord.model_validate(
        {
            "spec_id": "modeling__chose_greedy_over_dp__01",
            "target_weakness": {
                "fault_type": "modeling",
                "sub_tags": ["chose_greedy_over_dp"],
                "root_cause": "Cannot identify optimal substructure; defaults to greedy",
                "capability_cliff": "Greedy diverges from optimal when n>20",
            },
            "problem_spec": {
                "algorithm_type": "dynamic_programming",
                "difficulty": "hard",
                "narrative_theme": "logistics_scheduling",
                "constraints": {
                    "n_range": [1, 100000],
                    "value_range": [1, 1000000000],
                    "time_limit": "2s",
                    "memory_limit": "256MB",
                },
                "key_trap": "Greedy works on small examples but fails later.",
                "must_cover": [
                    "2D state transition",
                    "Overlapping subproblems",
                ],
                "must_avoid": [
                    "Must not be classic 0-1 knapsack",
                ],
            },
            "verification_spec": {
                "min_test_cases": 10,
                "must_include_edge_cases": [
                    "n=1 single element",
                    "n=100000 max scale stress test",
                ],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(2^n), n<=20",
            },
            "diversity_tags": {
                "narrative_theme": "logistics_scheduling",
                "data_structure": "array",
                "constraint_scale": "large",
            },
            "generation_hints": {
                "solution_approach": "Use 2D DP.",
                "common_wrong_approach": "Use greedy.",
                "distinguishing_test": "Three-item case where greedy loses.",
            },
            "language_constraint": {
                "target_languages": ["python", "cpp"],
                "language_specific": False,
            },
            "prompt_version": "v1",
        }
    )

    assert SpecRecord.model_validate_json(spec.model_dump_json()) == spec


def test_spec_rejects_malformed_ranges() -> None:
    with pytest.raises(ValueError, match="n_range|value_range"):
        SpecRecord.model_validate(
            {
                "spec_id": "modeling__chose_greedy_over_dp__01",
                "target_weakness": {
                    "fault_type": "modeling",
                    "sub_tags": ["chose_greedy_over_dp"],
                    "root_cause": "Cannot identify optimal substructure; defaults to greedy",
                    "capability_cliff": "Greedy diverges from optimal when n>20",
                },
                "problem_spec": {
                    "algorithm_type": "dynamic_programming",
                    "difficulty": "hard",
                    "narrative_theme": "logistics_scheduling",
                    "constraints": {
                        "n_range": [1, 2, 3],
                        "value_range": [1],
                        "time_limit": "2s",
                        "memory_limit": "256MB",
                    },
                    "key_trap": "Greedy works on small examples but fails later.",
                    "must_cover": ["2D state transition"],
                    "must_avoid": ["Must not be classic 0-1 knapsack"],
                },
                "verification_spec": {
                    "min_test_cases": 10,
                    "must_include_edge_cases": ["n=1 single element"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(2^n), n<=20",
                },
                "diversity_tags": {
                    "narrative_theme": "logistics_scheduling",
                    "data_structure": "array",
                    "constraint_scale": "large",
                },
                "generation_hints": {
                    "solution_approach": "Use 2D DP.",
                    "common_wrong_approach": "Use greedy.",
                    "distinguishing_test": "Three-item case where greedy loses.",
                },
                "language_constraint": {
                    "target_languages": ["python", "cpp"],
                    "language_specific": False,
                },
                "prompt_version": "v1",
            }
        )


def test_spec_rejects_non_positive_min_test_cases() -> None:
    with pytest.raises(ValueError, match="min_test_cases"):
        SpecRecord.model_validate(
            {
                "spec_id": "modeling__chose_greedy_over_dp__01",
                "target_weakness": {
                    "fault_type": "modeling",
                    "sub_tags": ["chose_greedy_over_dp"],
                    "root_cause": "Cannot identify optimal substructure; defaults to greedy",
                    "capability_cliff": "Greedy diverges from optimal when n>20",
                },
                "problem_spec": {
                    "algorithm_type": "dynamic_programming",
                    "difficulty": "hard",
                    "narrative_theme": "logistics_scheduling",
                    "constraints": {
                        "n_range": [1, 100000],
                        "value_range": [1, 1000000000],
                        "time_limit": "2s",
                        "memory_limit": "256MB",
                    },
                    "key_trap": "Greedy works on small examples but fails later.",
                    "must_cover": ["2D state transition"],
                    "must_avoid": ["Must not be classic 0-1 knapsack"],
                },
                "verification_spec": {
                    "min_test_cases": 0,
                    "must_include_edge_cases": ["n=1 single element"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(2^n), n<=20",
                },
                "diversity_tags": {
                    "narrative_theme": "logistics_scheduling",
                    "data_structure": "array",
                    "constraint_scale": "large",
                },
                "generation_hints": {
                    "solution_approach": "Use 2D DP.",
                    "common_wrong_approach": "Use greedy.",
                    "distinguishing_test": "Three-item case where greedy loses.",
                },
                "language_constraint": {
                    "target_languages": ["python", "cpp"],
                    "language_specific": False,
                },
                "prompt_version": "v1",
            }
        )


def test_run_metadata_round_trip() -> None:
    metadata = RunMetadata.model_validate(
        {
            "run_id": "20260416_143022_a3f1",
            "timestamp": "2026-04-16T14:30:22Z",
            "config_snapshot": {"model": {"analysis_model": "claude-sonnet-4-20250514"}},
            "analysis_model": "claude-sonnet-4-20250514",
            "prompt_versions": {
                "diagnose": "v2",
                "aggregate": "v3",
                "synthesize": "v1",
            },
            "input_files": ["eval_log.jsonl"],
            "input_count": 1000,
            "stages_executed": ["diagnose", "aggregate", "synthesize"],
            "self_analysis_warning": False,
            "summary": {
                "diagnosed": 1000,
                "rule_screened": 200,
                "model_analyzed": 800,
                "non_failures": 0,
                "errors": 3,
                "skipped": 0,
                "elapsed_seconds": 12.5,
                "weaknesses_found": 12,
                "specs_generated": 36,
                "synthesize_failures": 2,
                "specs_by_weakness": {"state_tracking": 24, "off_by_one": 12},
                "synthesize_status": "success",
                "attempted_weaknesses": ["state_tracking", "off_by_one"],
                "covered_weaknesses": ["state_tracking", "off_by_one"],
                "weaknesses_without_specs": [],
                "synthesize_failure_reasons_by_weakness": {},
            },
        }
    )

    assert RunMetadata.model_validate_json(metadata.model_dump_json()) == metadata


def test_run_metadata_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        RunMetadata.model_validate(
            {
                "run_id": "20260416_143022_a3f1",
                "timestamp": "2026-04-16T14:30:22Z",
                "config_snapshot": {"model": {"analysis_model": "claude-sonnet-4-20250514"}},
                "analysis_model": "claude-sonnet-4-20250514",
                "prompt_versions": {
                    "diagnose": "v2",
                    "aggregate": "v3",
                    "synthesize": "v1",
                    "unexpected": "extra",
                },
                "input_files": ["eval_log.jsonl"],
                "input_count": 1000,
                "stages_executed": ["diagnose", "aggregate", "synthesize"],
                "summary": {
                    "diagnosed": 1000,
                    "rule_screened": 200,
                    "model_analyzed": 800,
                    "non_failures": 0,
                    "errors": 3,
                    "skipped": 0,
                    "elapsed_seconds": 12.5,
                    "weaknesses_found": 12,
                    "specs_generated": 36,
                    "synthesize_failures": 2,
                    "specs_by_weakness": {"state_tracking": 24, "off_by_one": 12},
                    "synthesize_status": "success",
                    "weaknesses_without_specs": [],
                },
            }
        )
