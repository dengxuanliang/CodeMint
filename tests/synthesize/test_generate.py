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


def test_key_trap_with_unquoted_but_specific_evidence_reference_passes() -> None:
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
                "The trap only passes if the solver keeps the final endpoint included "
                "and avoids dropping the terminal index after each expansion."
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

    assert "final endpoint included" in spec.problem_spec.key_trap


def test_key_trap_with_specific_name_error_reference_passes() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["function_name_mismatch"],
        frequency=2,
        sample_task_ids=[103],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Function name mismatches break execution.",
            capability_cliff="Strict test harnesses require exact entry-point names.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.9,
        ),
    )
    original_evidence = {
        "wrong_line": "def solve_value(x):",
        "correct_approach": "Define the function with the exact name expected by the test harness, def solve(x):.",
        "failed_test": "NameError: name 'solve' is not defined",
    }

    spec = generate_spec(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        invoke_model=lambda prompt: {
            "algorithm_type": "dynamic programming",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": (
                "The trap fails immediately if the solver exposes solve_value instead of solve, "
                "triggering the same NameError seen in the original evidence."
            ),
            "must_cover": ["function_name_mismatch"],
            "must_avoid": ["renaming the required entry point"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Implement the exact solve(x) entry point.",
                "common_wrong_approach": "Use solve_value(x) instead of solve(x).",
                "distinguishing_test": "Call solve() directly from the harness.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        },
        original_evidence=original_evidence,
        spec_index=1,
    )

    assert "solve_value" in spec.problem_spec.key_trap
    assert "solve" in spec.problem_spec.key_trap


def test_function_name_mismatch_grounding_accepts_entry_point_name_reference() -> None:
    from codemint.synthesize.generate import has_concrete_evidence_reference

    original_evidence = {
        "wrong_line": "```python\ndef solve_value(x):\n    return x * 2\n```",
        "correct_approach": "Inspect the failing behavior in context and reason from tests.",
        "failed_test": '{"code": "assert solve(2) == 4"}',
    }

    assert has_concrete_evidence_reference(
        "The trap fails when solve_value is exposed instead of solve.",
        original_evidence,
    )


def test_missing_code_block_grounding_accepts_missing_code_phrase_reference() -> None:
    from codemint.synthesize.generate import has_concrete_evidence_reference

    original_evidence = {
        "wrong_line": "I would solve this by defining a function and returning x plus one, but the final code block is missing.",
        "correct_approach": "Inspect the failing behavior in context and reason from tests.",
        "failed_test": '{"code": "assert solve(1) == 2"}',
    }

    assert has_concrete_evidence_reference(
        "The trap reproduces the original final code block is missing failure instead of executable solve(x) output.",
        original_evidence,
    )


def test_missing_code_block_grounding_accepts_explanation_instead_of_code_reference() -> None:
    from codemint.synthesize.generate import has_concrete_evidence_reference

    original_evidence = {
        "wrong_line": "I would solve this by defining a function and returning x plus one, but the final code block is missing.",
        "correct_approach": "Inspect the failing behavior in context and reason from tests.",
        "failed_test": '{"code": "assert solve(1) == 2"}',
    }

    assert has_concrete_evidence_reference(
        "The trap fails when the solver explains the approach instead of returning a callable solve function.",
        original_evidence,
    )


def test_off_by_one_grounding_accepts_boundary_reference_from_real_evidence() -> None:
    from codemint.synthesize.generate import has_concrete_evidence_reference

    original_evidence = {
        "wrong_line": "```python\ndef solve(x)\n    return x * 2\n``` | ```python\ndef solve(x):\n    return x - 1\n```",
        "correct_approach": "Inspect the failing behavior in context and reason from tests.",
        "failed_test": '{"code": "assert solve(2) == 4"} | {"code": "assert solve(3) == 2"}',
    }

    assert has_concrete_evidence_reference(
        "The trap catches the boundary bug where solve returns x - 1 on the failing example.",
        original_evidence,
    )


def test_logic_error_grounding_accepts_wrong_formula_reference_from_real_evidence() -> None:
    from codemint.synthesize.generate import has_concrete_evidence_reference

    original_evidence = {
        "wrong_line": "```python\ndef solve(x)\n    return x * 2\n``` | ```python\ndef solve(x):\n    return x - 1\n```",
        "correct_approach": "Inspect the failing behavior in context and reason from tests.",
        "failed_test": '{"code": "assert solve(2) == 4"} | {"code": "assert solve(3) == 2"}',
    }

    assert has_concrete_evidence_reference(
        "The trap rejects arithmetic shortcuts like returning x * 2 or x - 1 instead of the required logic.",
        original_evidence,
    )


def test_function_name_mismatch_spec_requires_exact_entry_point_contract() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["function_name_mismatch"],
        frequency=2,
        sample_task_ids=[103],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Function name mismatches break execution.",
            capability_cliff="Strict test harnesses require exact entry-point names.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.9,
        ),
    )
    original_evidence = {
        "wrong_line": "def solve_value(x):",
        "correct_approach": "Define the function with the exact name expected by the test harness, def solve(x):.",
        "failed_test": "NameError: name 'solve' is not defined",
    }

    spec = generate_spec(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        invoke_model=lambda prompt: {
            "algorithm_type": "dynamic programming",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": (
                "The trap fails immediately if the solver exposes solve_value instead of solve, "
                "triggering the same NameError seen in the original evidence."
            ),
            "must_cover": ["exact callable entry point solve(x)", "single public function contract"],
            "must_avoid": ["alternative public function names", "renaming the required entry point"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Implement the exact solve(x) entry point.",
                "common_wrong_approach": "Use solve_value(x) instead of solve(x).",
                "distinguishing_test": "Call solve() directly from the harness.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        },
        original_evidence=original_evidence,
        spec_index=1,
    )

    assert any("exact callable entry point" in item for item in spec.problem_spec.must_cover)
    assert any("alternative public function names" in item for item in spec.problem_spec.must_avoid)


def test_missing_code_block_spec_gets_explicit_code_output_constraints() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["missing_code_block"],
        frequency=2,
        sample_task_ids=[101],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Explanation is returned instead of executable code.",
            capability_cliff="Direct code emission fails when the model drifts into prose.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.92,
        ),
    )
    original_evidence = {
        "wrong_line": "I would solve this by defining a function and returning x plus one, but the final code block is missing.",
        "correct_approach": "Generate the Python function definition def solve(x): return x + 1.",
        "failed_test": "Execution failed because no callable solve function was produced.",
    }

    spec = generate_spec(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        invoke_model=lambda prompt: {
            "algorithm_type": "simulation",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": "The trap fails when the solver explains the approach instead of returning a callable solve function.",
            "must_cover": ["basic implementation correctness"],
            "must_avoid": ["verbatim reuse of prior tasks"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Return the executable function implementation directly.",
                "common_wrong_approach": "Explain the intended code without emitting it.",
                "distinguishing_test": "Check that a callable solve function exists.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        },
        original_evidence=original_evidence,
        spec_index=1,
    )

    assert any("executable code" in item for item in spec.problem_spec.must_cover)
    assert any("explanation" in item for item in spec.problem_spec.must_avoid)


def test_markdown_formatting_spec_gets_raw_output_constraints() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = WeaknessEntry(
        rank=1,
        fault_type="surface",
        sub_tags=["markdown_formatting"],
        frequency=3,
        sample_task_ids=[201],
        trainability=0.3,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Markdown fences wrap otherwise executable code.",
            capability_cliff="The answer becomes unusable when fenced formatting leaks into raw output.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.91,
        ),
    )
    original_evidence = {
        "wrong_line": "```python\ndef solve(x):\n    return x + 1\n```",
        "correct_approach": "Return raw executable code without markdown fences.",
        "failed_test": "SyntaxError: invalid syntax",
    }

    spec = generate_spec(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        invoke_model=lambda prompt: {
            "algorithm_type": "simulation",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
                "key_trap": "The trap fails when the model wraps the original ````python` fenced solution in markdown fences instead of returning raw code.",
            "must_cover": ["raw executable code output", "no markdown fences"],
            "must_avoid": ["wrapping delimiters", "fenced code blocks"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Return the raw solve(x) implementation only.",
                "common_wrong_approach": "Wrap the correct code in markdown fences.",
                "distinguishing_test": "Reject output if it contains fences or backticks.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        },
        original_evidence=original_evidence,
        spec_index=1,
    )

    assert any("raw executable code" in item for item in spec.problem_spec.must_cover)
    assert any("fenced code blocks" in item for item in spec.problem_spec.must_avoid)


def test_syntax_error_spec_gets_syntactic_completeness_constraints() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["syntax_error"],
        frequency=1,
        sample_task_ids=[102],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="The code is syntactically invalid.",
            capability_cliff="Execution fails before semantic correctness can be tested.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.93,
        ),
    )
    original_evidence = {
        "wrong_line": "def solve(x)",
        "correct_approach": "Add a colon at the end of the function definition: def solve(x):",
        "failed_test": "SyntaxError: expected ':'",
    }

    spec = generate_spec(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        invoke_model=lambda prompt: {
            "algorithm_type": "simulation",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": "The trap reproduces the original def solve(x) header without the required colon.",
            "must_cover": ["basic implementation correctness"],
            "must_avoid": ["verbatim reuse of prior tasks"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Return syntactically valid executable code.",
                "common_wrong_approach": "Omit the colon after the function header.",
                "distinguishing_test": "Parse the final code before execution.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        },
        original_evidence=original_evidence,
        spec_index=1,
    )

    assert any("syntactically complete executable code" in item for item in spec.problem_spec.must_cover)
    assert any("missing colons" in item or "missing bodies" in item for item in spec.problem_spec.must_avoid)


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
