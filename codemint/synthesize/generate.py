from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from codemint.modeling.parser import _normalize_json_text
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
from codemint.synthesize.contract_normalizer import (
    describe_public_contract_target,
    describe_wrong_public_contract_target,
    normalize_contracts,
    resolve_function_name_mismatch_target,
)
from codemint.synthesize.input_view import build_synthesis_input_view


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
    repair_context: dict[str, str] | None = None,
) -> SpecRecord:
    prompt = load_prompt("synthesize_spec_generation")
    input_view = build_synthesis_input_view(weakness, original_evidence)
    inferred_language_constraint = _inferred_language_constraint(input_view)
    payload = {
        "template": prompt.template,
        "weakness": {
            "fault_type": input_view.fault_type,
            "primary_sub_tag": input_view.primary_sub_tag,
            "frequency": input_view.frequency,
            "sample_task_ids": input_view.sample_task_ids,
            "canonical_summary": input_view.canonical_summary,
        },
        "original_evidence": input_view.representative_evidence,
        "language_profile": {
            "primary_language": input_view.primary_language,
            "target_languages": input_view.target_languages,
            "language_specific": input_view.language_specific,
        },
        "diversity_tags": diversity_tags.model_dump(mode="json"),
        "difficulty": difficulty,
        "must_avoid_constraints": must_avoid_constraints or [],
        "repair_context": repair_context or {"mode": "initial_generation", "reason": ""},
    }
    response = _validate_response(invoke_model(payload))
    response_language_constraint = _resolved_language_constraint(
        response.language_constraint,
        inferred_language_constraint,
    )
    assigned_difficulty = difficulty or response.difficulty
    key_trap = _require_evidence_grounding(response.key_trap, original_evidence)
    must_cover = _dedupe_strings(_augment_must_cover(weakness, response.must_cover))
    must_avoid = _dedupe_strings(_augment_must_avoid(weakness, response.must_avoid))
    final_must_avoid = _dedupe_strings(must_avoid + list(must_avoid_constraints or []))
    _validate_function_name_grounding(weakness, key_trap, must_cover, must_avoid, response.generation_hints, original_evidence)
    normalized_cover, normalized_avoid = normalize_contracts(
        weakness,
        must_cover=must_cover,
        must_avoid=final_must_avoid,
        target_languages=response_language_constraint.target_languages,
        context_texts=[
            key_trap,
            response.generation_hints.solution_approach,
            response.generation_hints.common_wrong_approach,
            response.generation_hints.distinguishing_test,
        ],
        original_evidence=original_evidence,
    )
    final_key_trap = key_trap
    final_generation_hints = response.generation_hints
    if "function_name_mismatch" in weakness.sub_tags:
        target = resolve_function_name_mismatch_target(
            must_cover=must_cover,
            context_texts=[
                key_trap,
                response.generation_hints.solution_approach,
                response.generation_hints.common_wrong_approach,
                response.generation_hints.distinguishing_test,
            ],
            original_evidence=original_evidence,
        )
        final_key_trap, final_generation_hints = _rewrite_function_name_mismatch_narrative(
            key_trap=key_trap,
            generation_hints=response.generation_hints,
            target_description=describe_public_contract_target(target),
            wrong_description=describe_wrong_public_contract_target(original_evidence),
        )

    return SpecRecord(
        spec_id=f"spec-{spec_index:04d}",
        target_weakness=TargetWeakness(
            fault_type=weakness.fault_type,
            sub_tags=weakness.sub_tags,
            root_cause=input_view.primary_sub_tag,
            capability_cliff=input_view.canonical_summary,
        ),
        problem_spec=ProblemSpec(
            algorithm_type=response.algorithm_type,
            difficulty=assigned_difficulty,
            narrative_theme=diversity_tags.narrative_theme,
            constraints=response.constraints,
            key_trap=final_key_trap,
            must_cover=normalized_cover,
            must_avoid=normalized_avoid,
        ),
        verification_spec=response.verification_spec,
        diversity_tags=diversity_tags,
        generation_hints=final_generation_hints,
        language_constraint=response_language_constraint,
        prompt_version=prompt.version,
    )


def default_invoke_model(payload: dict[str, Any]) -> dict[str, Any]:
    weakness = payload["weakness"]
    evidence = payload["original_evidence"]
    language_profile = payload["language_profile"]
    diversity = payload["diversity_tags"]
    difficulty = payload["difficulty"] or "medium"
    scale = diversity["constraint_scale"]

    return {
        "algorithm_type": _algorithm_type_for_fault(weakness["fault_type"]),
        "difficulty": difficulty,
        "narrative_theme": diversity["narrative_theme"],
        "constraints": _constraints_for_scale(scale),
        "key_trap": f"Reference the original evidence: {evidence['wrong_line']}",
        "must_cover": [weakness["primary_sub_tag"], evidence["correct_approach"]],
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
            "target_languages": language_profile["target_languages"]
            if language_profile["primary_language"] != "unknown"
            else ["python"],
            "language_specific": language_profile["language_specific"],
        },
    }


def _validate_response(raw: Any) -> GenerationResponse:
    payload = raw
    if isinstance(raw, str):
        payload = json.loads(_normalize_json_text(raw))
    payload = _normalize_generation_payload(payload)

    try:
        return GenerationResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Synthesize generation response did not match schema: {exc}") from exc


def parse_generation_response(raw: Any) -> GenerationResponse:
    return _validate_response(raw)


def _normalize_generation_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    normalized = {
        key: payload[key]
        for key in (
            "algorithm_type",
            "difficulty",
            "narrative_theme",
            "constraints",
            "key_trap",
            "must_cover",
            "must_avoid",
            "verification_spec",
            "generation_hints",
            "language_constraint",
        )
        if key in payload
    }
    normalized["must_cover"] = _normalize_string_list(payload.get("must_cover"))
    normalized["must_avoid"] = _normalize_string_list(payload.get("must_avoid"))
    normalized["constraints"] = _normalize_constraints(payload.get("constraints"))
    normalized["verification_spec"] = _normalize_verification_spec(payload.get("verification_spec"))
    normalized["generation_hints"] = _normalize_generation_hints(payload.get("generation_hints"))
    normalized["language_constraint"] = _normalize_language_constraint(payload.get("language_constraint"))
    return normalized


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
    if _contains_identifier_reference(candidate_text, wrong_line):
        return True
    if _contains_phrase_reference(candidate_text, wrong_line):
        return True
    if _contains_formula_reference(candidate_text, wrong_line):
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


def _contains_identifier_reference(candidate_text: str, evidence_value: str) -> bool:
    identifiers = set(re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", evidence_value))
    identifiers.update(token for token in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", evidence_value) if "_" in token)
    for identifier in identifiers:
        normalized = identifier.strip().lower()
        if len(normalized) >= 4 and normalized not in {"python", "return"} and normalized in candidate_text:
            return True
    return False


def _contains_phrase_reference(candidate_text: str, evidence_value: str) -> bool:
    phrases = [
        "final code block is missing",
        "code block is missing",
        "missing code block",
        "explanation instead of code",
        "explains the approach instead of returning",
        "explain the approach instead of returning",
        "returning a callable solve function",
        "callable solve function",
    ]
    normalized_evidence = evidence_value.lower()
    if any(phrase in normalized_evidence and phrase in candidate_text for phrase in phrases):
        return True
    if "final code block is missing" in normalized_evidence:
        return (
            ("explains the approach" in candidate_text or "explanation" in candidate_text)
            and ("callable solve function" in candidate_text or "returning code" in candidate_text)
        )
    return False


def _contains_formula_reference(candidate_text: str, evidence_value: str) -> bool:
    formulas = [match.strip().lower() for match in re.findall(r"return\s+([^\n`|]+)", evidence_value)]
    if not formulas:
        return False

    referenced = 0
    for formula in formulas:
        normalized = formula.strip()
        if normalized and normalized in candidate_text:
            referenced += 1

    return referenced > 0


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


def _augment_must_cover(weakness: WeaknessEntry, must_cover: list[str]) -> list[str]:
    augmented = list(must_cover)
    if "function_name_mismatch" in weakness.sub_tags:
        target_entrypoint = _target_entrypoint_from_evidence(weakness, augmented)
        _append_if_missing(
            augmented,
            f"Require a single exact callable public entry point named {target_entrypoint}.",
        )
    if "missing_code_block" in weakness.sub_tags:
        target_entrypoint = _target_entrypoint_from_evidence(weakness, augmented)
        if _is_generic_entrypoint_placeholder(target_entrypoint):
            _append_if_missing(
                augmented,
                "Require executable code output with the requested callable entry point, not prose or explanation.",
            )
        else:
            _append_if_missing(
                augmented,
                f"Require executable code output with the requested callable entry point {target_entrypoint}, not prose or explanation.",
            )
    if "markdown_formatting" in weakness.sub_tags:
        _append_if_missing(
            augmented,
            "Require raw executable code output without markdown fences or wrapping delimiters.",
        )
    if "syntax_error" in weakness.sub_tags:
        target_entrypoint = _target_entrypoint_from_evidence(weakness, augmented)
        if _is_generic_entrypoint_placeholder(target_entrypoint):
            _append_if_missing(
                augmented,
                "Require syntactically complete executable code with a valid callable entry point definition.",
            )
        else:
            _append_if_missing(
                augmented,
                f"Require syntactically complete executable code with a valid {target_entrypoint} definition.",
            )
    if "non_executable_code" in weakness.sub_tags:
        _append_if_missing(
            augmented,
            "Require runnable code output with the requested executable implementation present.",
        )
    return augmented


def _augment_must_avoid(weakness: WeaknessEntry, must_avoid: list[str]) -> list[str]:
    augmented = list(must_avoid)
    if "function_name_mismatch" in weakness.sub_tags:
        _append_if_missing(
            augmented,
            "Do not allow alternate public function names such as solve_value, solver, or helper wrappers.",
        )
    if "missing_code_block" in weakness.sub_tags:
        _append_if_missing(
            augmented,
            "Do not replace the final code answer with explanation-only text or planning prose.",
        )
    if "markdown_formatting" in weakness.sub_tags:
        _append_if_missing(
            augmented,
            "Do not wrap the final answer in markdown fences, backticks, or other formatting delimiters.",
        )
    if "syntax_error" in weakness.sub_tags:
        _append_if_missing(
            augmented,
            "Do not emit incomplete code such as missing colons, missing bodies, or malformed function headers.",
        )
    if "non_executable_code" in weakness.sub_tags:
        _append_if_missing(
            augmented,
            "Do not return explanation-only, translation-only, prompt echo, or other prose instead of executable code.",
        )
    return augmented


def _append_if_missing(items: list[str], candidate: str) -> None:
    normalized_candidate = candidate.strip().lower()
    if not any(item.strip().lower() == normalized_candidate for item in items):
        items.append(candidate)


def _dedupe_strings(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _target_entrypoint_from_evidence(weakness: WeaknessEntry, existing_cover: list[str]) -> str:
    text = " ".join(
        [
            *existing_cover,
            weakness.collective_diagnosis.refined_root_cause,
            weakness.collective_diagnosis.capability_cliff,
        ]
    )
    identifiers = _function_names_in_text(text)
    if identifiers:
        return identifiers[-1]
    return "the requested entry point from the test harness"


def _is_generic_entrypoint_placeholder(name: str) -> bool:
    return name.strip().lower() == "the requested entry point from the test harness"


def _validate_function_name_grounding(
    weakness: WeaknessEntry,
    key_trap: str,
    must_cover: list[str],
    must_avoid: list[str],
    generation_hints: GenerationHints,
    original_evidence: dict[str, str],
) -> None:
    if "function_name_mismatch" not in weakness.sub_tags:
        return

    evidence_names = set(_function_names_in_text(" ".join(original_evidence.values())))
    candidate_text = " ".join(
        [
            key_trap,
            *must_cover,
            *must_avoid,
            generation_hints.solution_approach,
            generation_hints.common_wrong_approach,
            generation_hints.distinguishing_test,
        ]
    )
    candidate_names = set(_function_names_in_text(candidate_text))
    ungrounded = sorted(
        name
        for name in candidate_names
        if name not in evidence_names and name not in {"solve", "solver", "main", "helper"}
    )
    if ungrounded:
        raise ValueError(f"Generated spec contains ungrounded function name(s): {ungrounded}")


def _rewrite_function_name_mismatch_narrative(
    *,
    key_trap: str,
    generation_hints: GenerationHints,
    target_description: str,
    wrong_description: str,
) -> tuple[str, GenerationHints]:
    final_key_trap = (
        f"{key_trap.strip()} The task must enforce {target_description} and reject {wrong_description}."
    ).strip()
    final_hints = GenerationHints(
        solution_approach=f"Implement {target_description}.",
        common_wrong_approach=f"Expose a different public function name or public output shape than {target_description}.",
        distinguishing_test=f"Call only {target_description} and reject alternate public names or output shapes.",
    )
    return final_key_trap, final_hints


def _function_names_in_text(text: str) -> list[str]:
    names: list[str] = []
    patterns = [
        r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bfunction\s+named\s+['`\"]?([A-Za-z_][A-Za-z0-9_]*)",
        r"\bnamed\s+['`\"]([A-Za-z_][A-Za-z0-9_]*)",
        r"\bentry point\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"NameError:\s+name\s+['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            name = str(match)
            if name not in names and name not in {"function", "return", "assert"}:
                names.append(name)
    return names


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [
            item.strip()
            for item in re.split(r"[;\n,|]+", value)
            if item.strip()
        ]
    return []


def _normalize_constraints(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value

    numbers = [int(match) for match in re.findall(r"\d+", value)]
    n_range = [numbers[0], numbers[1]] if len(numbers) >= 2 else [1, 10_000]
    value_range = [numbers[2], numbers[3]] if len(numbers) >= 4 else [0, 1_000_000]
    time_match = re.search(r"(\d+\s*[smh]|[\d.]+s)", value, flags=re.IGNORECASE)
    memory_match = re.search(r"(\d+\s*(?:mb|gb))", value, flags=re.IGNORECASE)
    return {
        "n_range": n_range,
        "value_range": value_range,
        "time_limit": time_match.group(1).replace(" ", "") if time_match else "1s",
        "memory_limit": memory_match.group(1).replace(" ", "").upper() if memory_match else "256MB",
    }


def _normalize_verification_spec(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value

    min_test_match = re.search(r"at least\s+(\d+)\s+tests?", value, flags=re.IGNORECASE)
    edge_case = ""
    edge_match = re.search(r"edge case:\s*(.+?)(?:\.|$)", value, flags=re.IGNORECASE)
    if edge_match:
        edge_case = edge_match.group(1).strip()
    complexity_match = re.search(r"(O\([^)]+\))", value)
    return {
        "min_test_cases": int(min_test_match.group(1)) if min_test_match else 4,
        "must_include_edge_cases": [edge_case] if edge_case else [],
        "brute_force_verifiable": "yes" in value.lower() or "true" in value.lower(),
        "brute_force_complexity_limit": complexity_match.group(1) if complexity_match else "O(n^2)",
    }


def _normalize_generation_hints(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value

    solution = _extract_labeled_segment(value, "solution approach")
    wrong = _extract_labeled_segment(value, "common wrong approach")
    test = _extract_labeled_segment(value, "distinguishing test")
    return {
        "solution_approach": solution or value.strip(),
        "common_wrong_approach": wrong or value.strip(),
        "distinguishing_test": test or value.strip(),
    }


def _normalize_language_constraint(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value

    lowered = value.lower()
    targets: list[str] = []
    language_patterns = [
        ("python", r"\bpython\b"),
        ("cpp", r"\bcpp\b|\bc\+\+\b"),
        ("javascript", r"\bjavascript\b"),
        ("typescript", r"\btypescript\b"),
        ("java", r"\bjava\b"),
        ("go", r"\bgo\b"),
        ("rust", r"\brust\b"),
        ("r", r"\br\b"),
    ]
    for language, pattern in language_patterns:
        if re.search(pattern, lowered) and language not in targets:
            targets.append(language)
    return {
        "target_languages": targets or ["python"],
        "language_specific": len(targets) == 1,
    }


def _inferred_language_constraint(input_view) -> LanguageConstraint:
    if input_view.primary_language == "unknown":
        return LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        )

    return LanguageConstraint(
        target_languages=list(input_view.target_languages),
        language_specific=input_view.language_specific,
    )


def _resolved_language_constraint(
    model_constraint: LanguageConstraint,
    inferred_constraint: LanguageConstraint,
) -> LanguageConstraint:
    inferred_languages = set(inferred_constraint.target_languages)
    if inferred_constraint.target_languages == ["python"] and not inferred_constraint.language_specific:
        return model_constraint

    model_languages = set(model_constraint.target_languages)
    if model_languages and inferred_languages.issubset(model_languages):
        return model_constraint

    if model_languages:
        return inferred_constraint

    return inferred_constraint


def _extract_labeled_segment(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}:\s*(.+?)(?=(?:solution approach|common wrong approach|distinguishing test):|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


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
