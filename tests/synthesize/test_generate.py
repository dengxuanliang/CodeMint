from __future__ import annotations

import pytest

from codemint.models.spec import DiversityTags, SpecRecord
from codemint.models.weakness import CollectiveDiagnosis, WeaknessEntry


def test_generic_key_trap_without_concrete_evidence_reference_is_rejected() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = _weakness()
    original_evidence = _original_evidence()

    with pytest.raises(ValueError, match="key_trap must reference original evidence"):
        generate_spec(
            weakness,
            diversity_tags=DiversityTags(
                narrative_theme="sensors",
                data_structure="array",
                constraint_scale="medium",
            ),
            invoke_model=lambda prompt: {
                "algorithm_type": "prefix sums",
                "difficulty": "medium",
                "narrative_theme": "sensors",
                "constraints": {
                    "n_range": [1, 5000],
                    "value_range": [0, 1000],
                    "time_limit": "1s",
                    "memory_limit": "256MB",
                },
                "key_trap": "This task should reference the original evidence and punish the same mistake.",
                "must_cover": ["off_by_one", "boundary updates"],
                "must_avoid": ["sorting"],
                "verification_spec": {
                    "min_test_cases": 4,
                    "must_include_edge_cases": ["single element", "last segment"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(n^2)",
                },
                "generation_hints": {
                    "solution_approach": "Track prefix totals and compare window endpoints.",
                    "common_wrong_approach": "Shift the right pointer before evaluating the current window.",
                    "distinguishing_test": "A valid window ending at the final index",
                },
                "language_constraint": {
                    "target_languages": ["python", "cpp"],
                    "language_specific": False,
                },
            },
            original_evidence=original_evidence,
            spec_index=1,
        )


def test_key_trap_using_only_failed_test_words_is_rejected() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = _weakness()
    original_evidence = _original_evidence()

    with pytest.raises(ValueError, match="key_trap must reference original evidence"):
        generate_spec(
            weakness,
            diversity_tags=DiversityTags(
                narrative_theme="sensors",
                data_structure="array",
                constraint_scale="medium",
            ),
            invoke_model=lambda prompt: {
                "algorithm_type": "prefix sums",
                "difficulty": "medium",
                "narrative_theme": "sensors",
                "constraints": {
                    "n_range": [1, 5000],
                    "value_range": [0, 1000],
                    "time_limit": "1s",
                    "memory_limit": "256MB",
                },
                "key_trap": "The trap is a last segment ending at index n-1.",
                "must_cover": ["off_by_one", "boundary updates"],
                "must_avoid": ["sorting"],
                "verification_spec": {
                    "min_test_cases": 4,
                    "must_include_edge_cases": ["single element", "last segment"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(n^2)",
                },
                "generation_hints": {
                    "solution_approach": "Track prefix totals and compare window endpoints.",
                    "common_wrong_approach": "Shift the right pointer before evaluating the current window.",
                    "distinguishing_test": "A valid window ending at the final index",
                },
                "language_constraint": {
                    "target_languages": ["python", "cpp"],
                    "language_specific": False,
                },
            },
            original_evidence=original_evidence,
            spec_index=1,
        )


def test_key_trap_grounded_across_wrong_line_and_correct_approach_passes() -> None:
    from codemint.synthesize.feasibility import check_feasibility
    from codemint.synthesize.generate import generate_spec

    weakness = _weakness()
    original_evidence = _original_evidence()

    spec = generate_spec(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="sensors",
            data_structure="array",
            constraint_scale="medium",
        ),
        invoke_model=lambda prompt: {
            "algorithm_type": "prefix sums",
            "difficulty": "medium",
            "narrative_theme": "sensors",
            "constraints": {
                "n_range": [1, 5000],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": (
                "The trap keeps `range(l, r)` in place and only passes when the solver "
                "checks the terminal index after each expansion instead of skipping it."
            ),
            "must_cover": ["off_by_one", "boundary updates"],
            "must_avoid": ["sorting"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single element", "last segment"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Track prefix totals and compare window endpoints.",
                "common_wrong_approach": "Shift the right pointer before evaluating the current window.",
                "distinguishing_test": "A valid window ending at the final index",
            },
            "language_constraint": {
                "target_languages": ["python", "cpp"],
                "language_specific": False,
            },
        },
        original_evidence=original_evidence,
        spec_index=1,
    )

    feasibility = check_feasibility(spec, original_evidence=original_evidence)

    assert isinstance(spec, SpecRecord)
    assert "`range(l, r)`" in spec.problem_spec.key_trap
    assert "checks the terminal index" in spec.problem_spec.key_trap
    assert spec.prompt_version == "v1"
    assert feasibility.accepted is True


def _weakness() -> WeaknessEntry:
    return WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["off_by_one"],
        frequency=4,
        sample_task_ids=[101, 102],
        trainability=0.7,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Boundary updates lag behind loop state.",
            capability_cliff="Inclusive and exclusive ranges are mixed at the end of the scan.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.84,
        ),
    )


def _original_evidence() -> dict[str, str]:
    return {
        "wrong_line": "Used `for i in range(l, r)` when the final endpoint should be included.",
        "correct_approach": "Check the terminal index after each expansion.",
        "failed_test": "Segment ending at index n-1 was skipped.",
    }
