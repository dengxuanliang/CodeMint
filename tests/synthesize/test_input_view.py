from __future__ import annotations

from codemint.models.weakness import CollectiveDiagnosis, WeaknessEntry


def test_build_synthesis_input_view_for_missing_code_block() -> None:
    from codemint.synthesize.input_view import build_synthesis_input_view

    weakness = _weakness("missing_code_block")
    evidence = {
        "wrong_line": "I would solve this by defining solve(x), but the final code block is missing.",
        "correct_approach": "Return executable solve(x) code directly.",
        "failed_test": "assert solve(1) == 2",
    }

    view = build_synthesis_input_view(weakness, evidence)

    assert view.fault_type == "implementation"
    assert view.primary_sub_tag == "missing_code_block"
    assert view.frequency == 2
    assert view.sample_task_ids == [101, 102]
    assert view.canonical_summary == "Model outputs explanation or transformed prompt instead of executable code."
    assert view.representative_evidence == evidence


def test_logic_error_summary_bucket_uses_output_contract_mismatch() -> None:
    from codemint.synthesize.input_view import build_synthesis_input_view

    weakness = _weakness("logic_error", fault_type="modeling")
    evidence = {
        "wrong_line": "Return only the count instead of the required sorted list.",
        "correct_approach": "Return the reordered list exactly as required by the prompt.",
        "failed_test": "assert candidate([3, 2, 1]) == [1, 2, 3]",
    }

    view = build_synthesis_input_view(weakness, evidence)

    assert view.primary_sub_tag == "logic_error"
    assert view.canonical_summary == "Model returns an output that violates the expected contract."


def test_logic_error_summary_bucket_uses_api_or_data_contract_misuse() -> None:
    from codemint.synthesize.input_view import build_synthesis_input_view

    weakness = _weakness("logic_error", fault_type="modeling")
    evidence = {
        "wrong_line": "Mutate the input map in place and treat missing keys as present.",
        "correct_approach": "Read the mapping without mutating it and handle missing keys safely.",
        "failed_test": "KeyError: 'missing'",
    }

    view = build_synthesis_input_view(weakness, evidence)

    assert view.canonical_summary == "Model misuses an API or data contract in executable code."


def test_build_synthesis_input_view_infers_python_language_profile() -> None:
    from codemint.synthesize.input_view import build_synthesis_input_view

    weakness = _weakness("missing_code_block")
    evidence = {
        "wrong_line": "```python\ndef solve(x):\n    return x + 1\n```",
        "correct_approach": "Return executable Python code.",
        "failed_test": "assert solve(1) == 2",
    }

    view = build_synthesis_input_view(weakness, evidence)

    assert view.primary_language == "python"
    assert view.target_languages == ["python"]
    assert view.language_specific is True


def test_build_synthesis_input_view_infers_r_language_profile() -> None:
    from codemint.synthesize.input_view import build_synthesis_input_view

    weakness = _weakness("markdown_formatting")
    evidence = {
        "wrong_line": "```r\nsolve <- function(x) x + 1\n```",
        "correct_approach": "Return executable R code.",
        "failed_test": "stopifnot(solve(1) == 2)",
    }

    view = build_synthesis_input_view(weakness, evidence)

    assert view.primary_language == "r"
    assert view.target_languages == ["r"]
    assert view.language_specific is True


def test_build_synthesis_input_view_uses_unknown_when_language_cannot_be_inferred() -> None:
    from codemint.synthesize.input_view import build_synthesis_input_view

    weakness = _weakness("non_executable_code")
    evidence = {
        "wrong_line": "Provide the answer directly without runnable code.",
        "correct_approach": "Return runnable code matching the task contract.",
        "failed_test": "The output is descriptive prose instead of code.",
    }

    view = build_synthesis_input_view(weakness, evidence)

    assert view.primary_language == "unknown"
    assert view.target_languages == ["unknown"]
    assert view.language_specific is False


def _weakness(tag: str, *, fault_type: str = "implementation") -> WeaknessEntry:
    return WeaknessEntry(
        rank=1,
        fault_type=fault_type,
        sub_tags=[tag],
        frequency=2,
        sample_task_ids=[101, 102],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause=f"{tag} root cause",
            capability_cliff=f"{tag} cliff",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.9,
        ),
    )
