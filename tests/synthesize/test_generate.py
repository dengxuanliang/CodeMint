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


def test_function_name_mismatch_constraints_use_evidence_entrypoint_not_solve() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["function_name_mismatch"],
        frequency=1,
        sample_task_ids=[234],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Function entry point name mismatch.",
            capability_cliff="Harness cannot call the generated function.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.95,
        ),
    )

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
            "key_trap": "The trap repeats the original wrong callable `compute_quantiles` instead of `get_column_quantiles`.",
            "must_cover": ["exact callable entry point get_column_quantiles(csv_path, quantiles)"],
            "must_avoid": ["alternate public function names such as compute_quantiles"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single csv file"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Expose get_column_quantiles.",
                "common_wrong_approach": "Expose compute_quantiles.",
                "distinguishing_test": "Call get_column_quantiles directly.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        },
        original_evidence={
            "wrong_line": "def compute_quantiles(csv_path, quantiles):",
            "correct_approach": "Define the exact get_column_quantiles(csv_path, quantiles) entry point expected by the harness.",
            "failed_test": "NameError: name 'get_column_quantiles' is not defined",
        },
        spec_index=1,
    )

    cover_text = " ".join(spec.problem_spec.must_cover)
    assert "get_column_quantiles" in cover_text
    assert "solve(x)" not in cover_text


def test_generated_spec_rejects_new_function_names_not_grounded_in_evidence() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["function_name_mismatch"],
        frequency=1,
        sample_task_ids=[234],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Function entry point name mismatch.",
            capability_cliff="Harness cannot call the generated function.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.95,
        ),
    )

    with pytest.raises(ValueError, match="ungrounded function name"):
        generate_spec(
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
                "key_trap": "The trap repeats the original wrong callable `compute_quantiles`.",
                "must_cover": ["define exactly one public function named calculate_warehouse_profit"],
                "must_avoid": ["alternate public function names such as compute_quantiles"],
                "verification_spec": {
                    "min_test_cases": 4,
                    "must_include_edge_cases": ["single csv file"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(n^2)",
                },
                "generation_hints": {
                    "solution_approach": "Expose calculate_warehouse_profit.",
                    "common_wrong_approach": "Expose compute_quantiles.",
                    "distinguishing_test": "Call calculate_warehouse_profit directly.",
                },
                "language_constraint": {
                    "target_languages": ["python"],
                    "language_specific": False,
                },
            },
            original_evidence={
                "wrong_line": "def compute_quantiles(csv_path, quantiles):",
                "correct_approach": "Define the exact get_column_quantiles(csv_path, quantiles) entry point expected by the harness.",
                "failed_test": "NameError: name 'get_column_quantiles' is not defined",
            },
            spec_index=1,
        )


def test_generate_spec_payload_uses_canonical_summary_not_aggregate_narrative() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = _weakness()
    seen: dict[str, object] = {}

    def invoke_model(payload):
        seen.update(payload)
        return {
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
        }

    spec = generate_spec(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="sensors",
            data_structure="array",
            constraint_scale="medium",
        ),
        invoke_model=invoke_model,
        original_evidence=_original_evidence(),
        spec_index=1,
    )

    weakness_payload = seen["weakness"]
    assert "root_cause" not in weakness_payload
    assert "capability_cliff" not in weakness_payload
    assert weakness_payload["primary_sub_tag"] == "off_by_one"
    assert weakness_payload["canonical_summary"] == "Model computes the wrong result with executable logic."
    assert spec.target_weakness.root_cause == "off_by_one"
    assert spec.target_weakness.capability_cliff == "Model computes the wrong result with executable logic."


def test_language_constraint_payload_includes_inferred_r_language_profile() -> None:
    from codemint.synthesize.generate import generate_spec

    seen: dict[str, object] = {}

    def invoke_model(payload):
        seen.update(payload)
        return _generation_response(target_languages=["r"], language_specific=True)

    generate_spec(
        _weakness(),
        diversity_tags=DiversityTags(
            narrative_theme="signals",
            data_structure="array",
            constraint_scale="medium",
        ),
        invoke_model=invoke_model,
        original_evidence=_r_original_evidence(),
        spec_index=1,
    )

    assert seen["language_profile"] == {
        "primary_language": "r",
        "target_languages": ["r"],
        "language_specific": True,
    }


def test_language_constraint_uses_inferred_r_language_with_default_model() -> None:
    from codemint.synthesize.generate import default_invoke_model, generate_spec

    spec = generate_spec(
        _weakness(),
        diversity_tags=DiversityTags(
            narrative_theme="signals",
            data_structure="array",
            constraint_scale="medium",
        ),
        invoke_model=default_invoke_model,
        original_evidence=_r_original_evidence(),
        spec_index=1,
    )

    assert spec.language_constraint.target_languages == ["r"]
    assert spec.language_constraint.language_specific is True


def test_language_constraint_uses_inferred_java_language_with_default_model() -> None:
    from codemint.synthesize.generate import default_invoke_model, generate_spec

    spec = generate_spec(
        _weakness(),
        diversity_tags=DiversityTags(
            narrative_theme="signals",
            data_structure="array",
            constraint_scale="medium",
        ),
        invoke_model=default_invoke_model,
        original_evidence=_java_original_evidence(),
        spec_index=1,
    )

    assert spec.language_constraint.target_languages == ["java"]
    assert spec.language_constraint.language_specific is True


def test_language_constraint_defaults_to_python_with_unknown_evidence() -> None:
    from codemint.synthesize.generate import default_invoke_model, generate_spec

    spec = generate_spec(
        _weakness(),
        diversity_tags=DiversityTags(
            narrative_theme="signals",
            data_structure="array",
            constraint_scale="medium",
        ),
        invoke_model=default_invoke_model,
        original_evidence=_unknown_language_evidence(),
        spec_index=1,
    )

    assert spec.language_constraint.target_languages == ["python"]
    assert spec.language_constraint.language_specific is False


def test_language_constraint_corrects_conflicting_model_output_for_known_language() -> None:
    from codemint.synthesize.generate import generate_spec

    spec = generate_spec(
        _weakness(),
        diversity_tags=DiversityTags(
            narrative_theme="signals",
            data_structure="array",
            constraint_scale="medium",
        ),
        invoke_model=lambda payload: _generation_response(
            target_languages=["python"],
            language_specific=False,
        ),
        original_evidence=_r_original_evidence(),
        spec_index=1,
    )

    assert spec.language_constraint.target_languages == ["r"]
    assert spec.language_constraint.language_specific is True


def test_language_constraint_preserves_broader_model_output_when_known_language_is_included() -> None:
    from codemint.synthesize.generate import generate_spec

    spec = generate_spec(
        _weakness(),
        diversity_tags=DiversityTags(
            narrative_theme="signals",
            data_structure="array",
            constraint_scale="medium",
        ),
        invoke_model=lambda payload: _generation_response(
            target_languages=["r", "python"],
            language_specific=False,
        ),
        original_evidence=_r_original_evidence(),
        spec_index=1,
    )

    assert spec.language_constraint.target_languages == ["r", "python"]
    assert spec.language_constraint.language_specific is False


def test_unknown_evidence_preserves_non_default_model_language_constraint() -> None:
    from codemint.synthesize.generate import generate_spec

    spec = generate_spec(
        _weakness(),
        diversity_tags=DiversityTags(
            narrative_theme="signals",
            data_structure="array",
            constraint_scale="medium",
        ),
        invoke_model=lambda payload: _generation_response(
            key_trap="The trap reproduces the original final code block is missing failure.",
            target_languages=["java", "python"],
            language_specific=False,
        ),
        original_evidence=_unknown_language_evidence(),
        spec_index=1,
    )

    assert spec.language_constraint.target_languages == ["java", "python"]
    assert spec.language_constraint.language_specific is False


@pytest.mark.parametrize(
    ("raw_value", "expected_targets"),
    [
        ("R", ["r"]),
        ("JavaScript", ["javascript"]),
        ("TypeScript", ["typescript"]),
        ("Go", ["go"]),
        ("Rust", ["rust"]),
        ("JavaScript and Python", ["python", "javascript"]),
    ],
)
def test_parse_generation_response_normalizes_string_language_constraint_variants(
    raw_value: str,
    expected_targets: list[str],
) -> None:
    from codemint.synthesize.generate import parse_generation_response

    response = parse_generation_response(
        {
            "algorithm_type": "simulation",
            "difficulty": "medium",
            "narrative_theme": "warehouse",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": "Reference `solve(x)` from the original evidence.",
            "must_cover": ["exact entry point"],
            "must_avoid": ["renamed entry point"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Implement the requested entry point directly.",
                "common_wrong_approach": "Rename the public callable.",
                "distinguishing_test": "Call the required function name from the harness.",
            },
            "language_constraint": raw_value,
        }
    )

    assert response.language_constraint.target_languages == expected_targets
    assert response.language_constraint.language_specific is (len(expected_targets) == 1)


def test_missing_code_block_recovers_entrypoint_from_aggregate_diagnosis_when_cover_is_generic() -> None:
    from codemint.synthesize.generate import generate_spec

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["missing_code_block"],
        frequency=2,
        sample_task_ids=[104],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="The required callable entry point is get_column_quantiles(csv_path, quantiles).",
            capability_cliff="The harness requires get_column_quantiles(csv_path, quantiles) in raw executable output.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.91,
        ),
    )
    original_evidence = {
        "wrong_line": "The answer explains the approach, but the final code block is missing.",
        "correct_approach": "Return the requested executable implementation directly.",
        "failed_test": "NameError: name 'get_column_quantiles' is not defined",
    }

    spec = generate_spec(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        invoke_model=lambda payload: {
            "algorithm_type": "simulation",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": "The trap reproduces the original final code block is missing failure instead of executable output.",
            "must_cover": ["basic implementation correctness"],
            "must_avoid": ["verbatim reuse of prior tasks"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Return the executable implementation directly.",
                "common_wrong_approach": "Explain the intended code without emitting it.",
                "distinguishing_test": "Check that the required callable exists.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        },
        original_evidence=original_evidence,
        spec_index=1,
    )

    assert any("get_column_quantiles" in item for item in spec.problem_spec.must_cover)


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


def _r_original_evidence() -> dict[str, str]:
    return {
        "wrong_line": "```r\nsolve <- function(x) {\n  x + 1\n}\n```",
        "correct_approach": "Return executable R code with solve <- function(x) { x + 1 }.",
        "failed_test": "The harness expected an R solve function.",
    }


def _java_original_evidence() -> dict[str, str]:
    return {
        "wrong_line": "public static int solve(int x) { return x + 1; }",
        "correct_approach": "Return executable Java code with public static int solve(int x).",
        "failed_test": "The Java harness called solve(1).",
    }


def _unknown_language_evidence() -> dict[str, str]:
    return {
        "wrong_line": "The final code block is missing from the answer.",
        "correct_approach": "Return the requested executable solution directly.",
        "failed_test": "No callable implementation was available for execution.",
    }


def _generation_response(
    *,
    target_languages: list[str],
    language_specific: bool,
    key_trap: str = "The trap reproduces `solve <- function(x)` from the original evidence.",
) -> dict[str, object]:
    return {
        "algorithm_type": "prefix sums",
        "difficulty": "medium",
        "narrative_theme": "signals",
        "constraints": {
            "n_range": [1, 5000],
            "value_range": [0, 1000],
            "time_limit": "1s",
            "memory_limit": "256MB",
        },
        "key_trap": key_trap,
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
            "target_languages": target_languages,
            "language_specific": language_specific,
        },
    }
