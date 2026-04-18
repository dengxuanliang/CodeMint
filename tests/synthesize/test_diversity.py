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


def _spec(spec_id: str, theme: str, data_structure: str, scale: str) -> SpecRecord:
    return SpecRecord(
        spec_id=spec_id,
        target_weakness=TargetWeakness(
            fault_type="modeling",
            sub_tags=["state_tracking"],
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
