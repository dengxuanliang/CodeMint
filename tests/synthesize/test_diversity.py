from __future__ import annotations

from codemint.config import CodeMintConfig
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
from codemint.models.weakness import CollectiveDiagnosis, WeaknessEntry


def test_diversity_assignment_rejects_overlap_above_threshold() -> None:
    from codemint.synthesize.diversity import assign_diversity_tags

    config = CodeMintConfig()
    existing = [
        _spec("spec-1", "graphs", "graph", "large"),
        _spec("spec-2", "graphs", "graph", "large"),
    ]

    result = assign_diversity_tags(
        existing,
        candidate_tags=DiversityTags(
            narrative_theme="graphs",
            data_structure="graph",
            constraint_scale="large",
        ),
        overlap_threshold=config.synthesize.diversity_overlap_threshold,
    )

    assert result.accepted is False
    assert result.overlap_score == 1.0
    assert "overlap" in result.reason


def test_plan_diversity_tags_only_considers_existing_specs_for_same_weakness() -> None:
    from codemint.synthesize.diversity import plan_diversity_tags

    config = CodeMintConfig.model_validate(
        {
            "synthesize": {
                "specs_per_weakness": 2,
                "max_per_weakness": 2,
                "top_n": 2,
                "narrative_themes": {
                    "generic": ["warehouses"],
                    "domain_adaptive": False,
                },
                "data_structures": ["array", "graph"],
            }
        }
    )
    target = _weakness("syntax_error")
    existing = [
        _spec("spec-1", "warehouses", "array", "small", weakness_tag="function_name_mismatch"),
        _spec("spec-2", "warehouses", "array", "medium", weakness_tag="missing_code"),
        _spec("spec-3", "warehouses", "array", "large", weakness_tag="formatting_error"),
    ]

    planned = plan_diversity_tags(target, 2, config.synthesize, existing)

    assert len(planned) == 2
    assert planned[0] != planned[1]

def test_planned_specs_default_to_python_when_language_is_unknown() -> None:
    from codemint.synthesize.diversity import _planned_specs

    planned = _planned_specs(
        [
            DiversityTags(
                narrative_theme="warehouses",
                data_structure="array",
                constraint_scale="small",
            )
        ],
        _weakness("syntax_error"),
    )

    assert planned[0].language_constraint.target_languages == ["python"]
    assert planned[0].language_constraint.language_specific is False


def test_planned_specs_use_inferred_r_language_when_weakness_context_is_r_specific() -> None:
    from codemint.synthesize.diversity import _planned_specs

    planned = _planned_specs(
        [
            DiversityTags(
                narrative_theme="warehouses",
                data_structure="array",
                constraint_scale="small",
            )
        ],
        _weakness(
            "markdown_formatting",
            root_cause="Omit the ```R and ``` markdown fences and output only raw executable R code.",
            capability_cliff="R source failed because markdown fences were present.",
        ),
    )

    assert planned[0].language_constraint.target_languages == ["r"]
    assert planned[0].language_constraint.language_specific is True


def test_planned_specs_use_inferred_java_language_when_weakness_context_is_java_specific() -> None:
    from codemint.synthesize.diversity import _planned_specs

    planned = _planned_specs(
        [
            DiversityTags(
                narrative_theme="warehouses",
                data_structure="array",
                constraint_scale="small",
            )
        ],
        _weakness(
            "missing_code_block",
            root_cause="Return executable Java code with public static int solve(int x).",
            capability_cliff="The Java harness expected a compilable solve(int) method.",
        ),
    )

    assert planned[0].language_constraint.target_languages == ["java"]
    assert planned[0].language_constraint.language_specific is True


def _spec(
    spec_id: str,
    theme: str,
    data_structure: str,
    scale: str,
    *,
    weakness_tag: str = "state_tracking",
) -> SpecRecord:
    return SpecRecord(
        spec_id=spec_id,
        target_weakness=TargetWeakness(
            fault_type="modeling",
            sub_tags=[weakness_tag],
            root_cause="state drift",
            capability_cliff="graph transitions",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="graph traversal",
            difficulty="hard",
            narrative_theme=theme,
            constraints=ProblemConstraints(
                n_range=[1, 200_000],
                value_range=[0, 10**9],
                time_limit="2s",
                memory_limit="256MB",
            ),
            key_trap="Track state across transitions.",
            must_cover=["state_tracking"],
            must_avoid=["verbatim reuse"],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single node"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme=theme,
            data_structure=data_structure,
            constraint_scale=scale,
        ),
        generation_hints=GenerationHints(
            solution_approach="Use BFS with augmented state.",
            common_wrong_approach="Greedy local transitions",
            distinguishing_test="Cycle that revisits a node with better state",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )


def _weakness(
    tag: str,
    *,
    root_cause: str | None = None,
    capability_cliff: str | None = None,
) -> WeaknessEntry:
    return WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=[tag],
        frequency=2,
        sample_task_ids=[1, 2],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause=root_cause or f"{tag} root cause",
            capability_cliff=capability_cliff or f"{tag} cliff",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.9,
        ),
    )
