from __future__ import annotations

from codemint.models.spec import DiversityTags, SpecRecord
from codemint.models.weakness import CollectiveDiagnosis, WeaknessEntry


def test_key_trap_must_reference_original_evidence() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = WeaknessEntry(
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
            "key_trap": "Forgetting the original failed evidence about inclusive end points.",
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
        original_evidence={
            "wrong_line": "Used `for i in range(l, r)` when the final endpoint should be included.",
            "correct_approach": "Check the terminal index after each expansion.",
            "failed_test": "Segment ending at index n-1 was skipped.",
        },
        spec_index=1,
    )

    assert isinstance(spec, SpecRecord)
    assert spec.problem_spec.key_trap.startswith("Forgetting the original failed evidence")
    assert spec.problem_spec.must_cover[:2] == ["off_by_one", "boundary updates"]
    assert spec.prompt_version == "v1"
