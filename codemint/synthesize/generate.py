from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from codemint.models.base import StrictModel
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
from codemint.models.weakness import WeaknessEntry
from codemint.prompts.registry import load_prompt


class GenerationResponse(StrictModel):
    algorithm_type: str
    difficulty: str
    narrative_theme: str
    constraints: ProblemConstraints
    key_trap: str
    must_cover: list[str]
    must_avoid: list[str]
    verification_spec: VerificationSpec
    generation_hints: GenerationHints
    language_constraint: LanguageConstraint


def generate_spec(
    weakness: WeaknessEntry,
    *,
    diversity_tags: DiversityTags,
    invoke_model,
    original_evidence: dict[str, str],
    spec_index: int,
    difficulty: str | None = None,
    must_avoid_constraints: list[str] | None = None,
) -> SpecRecord:
    prompt = load_prompt("synthesize_spec_generation")
    payload = {
        "template": prompt.template,
        "weakness": {
            "fault_type": weakness.fault_type,
            "sub_tags": weakness.sub_tags,
            "root_cause": weakness.collective_diagnosis.refined_root_cause,
            "capability_cliff": weakness.collective_diagnosis.capability_cliff,
        },
        "original_evidence": original_evidence,
        "diversity_tags": diversity_tags.model_dump(mode="json"),
        "difficulty": difficulty,
        "must_avoid_constraints": must_avoid_constraints or [],
    }
    response = _validate_response(invoke_model(payload))
    assigned_difficulty = difficulty or response.difficulty
    key_trap = _require_evidence_grounding(response.key_trap, original_evidence)

    return SpecRecord(
        spec_id=f"spec-{spec_index:04d}",
        target_weakness=TargetWeakness(
            fault_type=weakness.fault_type,
            sub_tags=weakness.sub_tags,
            root_cause=weakness.collective_diagnosis.refined_root_cause,
            capability_cliff=weakness.collective_diagnosis.capability_cliff,
        ),
        problem_spec=ProblemSpec(
            algorithm_type=response.algorithm_type,
            difficulty=assigned_difficulty,
            narrative_theme=diversity_tags.narrative_theme,
            constraints=response.constraints,
            key_trap=key_trap,
            must_cover=response.must_cover,
            must_avoid=response.must_avoid + list(must_avoid_constraints or []),
        ),
        verification_spec=response.verification_spec,
        diversity_tags=diversity_tags,
        generation_hints=response.generation_hints,
        language_constraint=response.language_constraint,
        prompt_version=prompt.version,
    )


def default_invoke_model(payload: dict[str, Any]) -> dict[str, Any]:
    weakness = payload["weakness"]
    evidence = payload["original_evidence"]
    diversity = payload["diversity_tags"]
    difficulty = payload["difficulty"] or "medium"
    scale = diversity["constraint_scale"]

    return {
        "algorithm_type": _algorithm_type_for_fault(weakness["fault_type"]),
        "difficulty": difficulty,
        "narrative_theme": diversity["narrative_theme"],
        "constraints": _constraints_for_scale(scale),
        "key_trap": f"Reference the original evidence: {evidence['wrong_line']}",
        "must_cover": [*weakness["sub_tags"][:2], evidence["correct_approach"]],
        "must_avoid": ["verbatim reuse of prior tasks", *payload["must_avoid_constraints"]],
        "verification_spec": {
            "min_test_cases": 4 if difficulty == "medium" else 5,
            "must_include_edge_cases": [evidence["failed_test"]],
            "brute_force_verifiable": True,
            "brute_force_complexity_limit": "O(n^2)",
        },
        "generation_hints": {
            "solution_approach": evidence["correct_approach"],
            "common_wrong_approach": evidence["wrong_line"],
            "distinguishing_test": evidence["failed_test"],
        },
        "language_constraint": {
            "target_languages": ["python"],
            "language_specific": False,
        },
    }


def _validate_response(raw: Any) -> GenerationResponse:
    payload = raw
    if isinstance(raw, str):
        payload = json.loads(raw)

    try:
        return GenerationResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Synthesize generation response did not match schema: {exc}") from exc


def _require_evidence_grounding(key_trap: str, original_evidence: dict[str, str]) -> str:
    if has_concrete_evidence_reference(key_trap, original_evidence):
        return key_trap

    raise ValueError("key_trap must reference original evidence")


def has_concrete_evidence_reference(text: str, original_evidence: dict[str, str]) -> bool:
    candidate_text = text.lower()
    wrong_line = original_evidence.get("wrong_line", "")
    correct_approach = original_evidence.get("correct_approach", "")

    if _contains_quoted_reference(candidate_text, wrong_line):
        return True
    if _contains_quoted_reference(candidate_text, correct_approach):
        return True
    if _contains_cross_field_reference(candidate_text, wrong_line, correct_approach):
        return True

    return False


def _contains_quoted_reference(candidate_text: str, evidence_value: str) -> bool:
    for quoted_span in re.findall(r"`([^`]+)`", evidence_value):
        normalized = quoted_span.strip().lower()
        if normalized and normalized in candidate_text:
            return True
    return False


def _contains_cross_field_reference(
    candidate_text: str,
    wrong_line: str,
    correct_approach: str,
) -> bool:
    wrong_tokens = _normalized_tokens(wrong_line)
    correct_tokens = _normalized_tokens(correct_approach)
    if not wrong_tokens or not correct_tokens:
        return False

    wrong_overlap = sum(1 for token in wrong_tokens if token in candidate_text)
    correct_overlap = sum(1 for token in correct_tokens if token in candidate_text)
    return wrong_overlap >= 1 and correct_overlap >= 1


def _normalized_tokens(text: str) -> list[str]:
    ignored = {
        "that",
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "then",
        "than",
        "but",
        "not",
        "only",
        "while",
        "into",
        "through",
        "before",
        "after",
        "when",
        "where",
        "unless",
        "instead",
        "each",
        "should",
        "with",
        "from",
        "used",
        "line",
        "test",
        "check",
        "segment",
        "original",
        "evidence",
        "failed",
        "task",
        "reference",
        "this",
        "same",
        "mistake",
        "punish",
        "index",
        "ending",
        "last",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9_\-]{3,}", text.lower())
        if token not in ignored
    ]


def _algorithm_type_for_fault(fault_type: str) -> str:
    mapping = {
        "modeling": "dynamic programming",
        "implementation": "simulation",
        "comprehension": "parsing",
        "edge_handling": "case analysis",
        "surface": "ad hoc",
    }
    return mapping.get(fault_type, "algorithmic reasoning")


def _constraints_for_scale(scale: str) -> dict[str, Any]:
    by_scale = {
        "small": {
            "n_range": [1, 100],
            "value_range": [0, 1000],
            "time_limit": "1s",
            "memory_limit": "128MB",
        },
        "medium": {
            "n_range": [1, 10_000],
            "value_range": [0, 1_000_000],
            "time_limit": "1s",
            "memory_limit": "256MB",
        },
        "large": {
            "n_range": [1, 1_000_000],
            "value_range": [0, 1_000_000_000],
            "time_limit": "2s",
            "memory_limit": "256MB",
        },
    }
    return by_scale.get(scale, by_scale["medium"])
