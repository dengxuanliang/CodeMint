from __future__ import annotations

from codemint.models.spec import (
    DiversityTags,
    GenerationHints,
    LanguageConstraint,
    ProblemConstraints,
    ProblemSpec,
    SpecRecord,
    TargetWeakness,
    VerificationSpec,
)


def test_extract_contract_signals_for_function_name_mismatch_semantic_variants() -> None:
    from codemint.synthesize.contracts import extract_contract_signals

    spec = _spec(
        weakness_tag="function_name_mismatch",
        must_cover=[
            "Require one public function named solve for the harness entry point.",
            "The checker must call the exact solver entrypoint directly.",
        ],
        must_avoid=[
            "Forbid helper entrypoint aliases like solve_value or solver.",
            "Do not rename the public callable.",
        ],
    )

    signals = extract_contract_signals(spec)

    assert signals.requires_exact_public_entry_point is True
    assert signals.forbids_alternate_public_names is True


def test_extract_contract_signals_for_missing_code_block_semantic_variants() -> None:
    from codemint.synthesize.contracts import extract_contract_signals

    spec = _spec(
        weakness_tag="missing_code_block",
        must_cover=[
            "Require emitted executable implementation output.",
            "The final answer must contain runnable code for the callable solve function.",
        ],
        must_avoid=[
            "Do not return explanation-only final responses.",
            "Avoid prose instead of executable code.",
        ],
    )

    signals = extract_contract_signals(spec)

    assert signals.requires_executable_code_output is True
    assert signals.forbids_explanation_only is True


def test_extract_contract_signals_for_markdown_formatting_semantic_variants() -> None:
    from codemint.synthesize.contracts import extract_contract_signals

    spec = _spec(
        weakness_tag="markdown_formatting",
        must_cover=[
            "Return plain executable program text only.",
            "The harness expects raw code output.",
        ],
        must_avoid=[
            "Do not wrap answers in markdown code fences.",
            "Avoid backticks or formatting delimiters around the final code.",
        ],
    )

    signals = extract_contract_signals(spec)

    assert signals.requires_raw_executable_output is True
    assert signals.forbids_markdown_wrapping is True


def test_extract_contract_signals_for_syntax_error_semantic_variants() -> None:
    from codemint.synthesize.contracts import extract_contract_signals

    spec = _spec(
        weakness_tag="syntax_error",
        must_cover=[
            "Require parseable and syntactically valid executable code.",
            "The emitted function definition must be complete.",
        ],
        must_avoid=[
            "Do not omit required punctuation like colons.",
            "Avoid malformed or partial function headers.",
        ],
    )

    signals = extract_contract_signals(spec)

    assert signals.requires_syntactic_completeness is True
    assert signals.forbids_incomplete_code is True


def _spec(*, weakness_tag: str, must_cover: list[str], must_avoid: list[str]) -> SpecRecord:
    return SpecRecord(
        spec_id="spec-0001",
        target_weakness=TargetWeakness(
            fault_type="implementation",
            sub_tags=[weakness_tag],
            root_cause="root cause",
            capability_cliff="capability cliff",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="simulation",
            difficulty="medium",
            narrative_theme="warehouses",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="trap references original evidence",
            must_cover=must_cover,
            must_avoid=must_avoid,
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single value input"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        generation_hints=GenerationHints(
            solution_approach="Do the right thing.",
            common_wrong_approach="Do the wrong thing.",
            distinguishing_test="Observe the difference.",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )
