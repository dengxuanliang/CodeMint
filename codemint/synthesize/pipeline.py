from __future__ import annotations

import json
from pathlib import Path

from codemint.config import CodeMintConfig
from codemint.io.jsonl import append_jsonl, read_jsonl
from codemint.modeling.client import ModelClient
from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.spec import SpecRecord
from codemint.prompts.registry import load_prompt
from codemint.models.weakness import WeaknessEntry, WeaknessReport
from codemint.synthesize.allocation import allocate_specs, select_top_weaknesses, weakness_key
from codemint.synthesize.diversity import assign_diversity_tags, plan_diversity_tags
from codemint.synthesize.feasibility import check_feasibility
from codemint.synthesize.generate import default_invoke_model, generate_spec, parse_generation_response
from codemint.synthesize.language_profile import infer_language_profile


def run_synthesize(
    report: WeaknessReport,
    output_path: Path,
    *,
    config: CodeMintConfig | None = None,
    existing_specs: list[SpecRecord] | None = None,
    invoke_model=None,
    feasibility_check=None,
    diagnoses: list[DiagnosisRecord] | None = None,
    original_evidence: dict[str, dict[str, str]] | None = None,
    progress_callback=None,
) -> list[SpecRecord]:
    resolved_config = config or CodeMintConfig()
    synthesize_config = resolved_config.synthesize
    allocated = allocate_specs(report, synthesize_config)
    generated: list[SpecRecord] = []
    prior_specs = list(existing_specs or [])
    model_invoke = invoke_model or _default_invoke_model(resolved_config)
    evidence_map = original_evidence or build_original_evidence_map(report, diagnoses or [])
    sample_evidence_map = build_sample_evidence_map(report, diagnoses or [])
    attempted_weaknesses: list[str] = []
    total_slots = 0
    slot_progress = 0
    selected_weaknesses = select_top_weaknesses(report.weaknesses, synthesize_config.top_n)

    for weakness in selected_weaknesses:
        key = weakness_key(weakness)
        attempted_weaknesses.append(key)
        total_slots += allocated.get(key, synthesize_config.specs_per_weakness)

    for weakness in selected_weaknesses:
        key = weakness_key(weakness)
        count = allocated.get(key, synthesize_config.specs_per_weakness)
        diversity_plan = plan_diversity_tags(
            weakness,
            count,
            synthesize_config,
            prior_specs + generated,
        )
        for slot_index, diversity_tags in enumerate(diversity_plan, start=1):
            spec = _generate_or_log_failure(
                weakness,
                output_path=output_path,
                diversity_tags=diversity_tags,
                spec_index=len(generated) + 1,
                difficulty=_difficulty_for_slot(slot_index - 1, count, synthesize_config.difficulty_distribution),
                invoke_model=model_invoke,
                feasibility_check=feasibility_check,
                original_evidence=_evidence_for_slot(
                    weakness,
                    evidence_map,
                    sample_evidence_map,
                    slot_index - 1,
                ),
                overlap_threshold=synthesize_config.diversity_overlap_threshold,
                existing_specs=prior_specs + generated,
                max_attempts=synthesize_config.max_regeneration_attempts + 1,
            )
            if spec is not None:
                generated.append(spec)
            slot_progress += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "synthesize",
                        "status": "running",
                        "processed": slot_progress,
                        "total": total_slots,
                        "errors": 0,
                        "eta_seconds": max(total_slots - slot_progress, 0) * 3,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    if generated:
        append_jsonl(output_path, [spec.model_dump(mode="json") for spec in generated])
    else:
        raise ValueError("No specs were synthesized successfully")
    return generated


def _default_invoke_model(config: CodeMintConfig):
    client = _build_model_client(config)
    if client is None:
        return default_invoke_model

    prompt = load_prompt("synthesize_spec_generation")

    def invoke(payload: dict):
        request_payload = {k: v for k, v in payload.items() if k != "template"}
        last_error: Exception | None = None

        def parse_invoke(format_error: str | None):
            user_prompt = f"{prompt.template}\n\nPayload JSON:\n{request_payload}"
            if format_error:
                user_prompt += f"\n\nFormat correction:\n{format_error}"
            return parse_generation_response(client.complete("Return only valid JSON.", str(user_prompt)))

        try:
            parsed = parse_invoke(None)
        except Exception as first_error:
            last_error = first_error
            parsed = parse_invoke(
                "Return JSON matching the exact required schema. "
                "Do not include tasks, task_specifications, prompt, or tags. "
                f"Error: {first_error}"
            )

        return parsed.model_dump(mode="json")

    return invoke


def _build_model_client(config: CodeMintConfig) -> ModelClient | None:
    model = config.model
    if not (model.analysis_model and model.base_url and model.api_key):
        return None
    return ModelClient(model)


def _generate_or_log_failure(
    weakness: WeaknessEntry,
    *,
    output_path: Path,
    diversity_tags,
    spec_index: int,
    difficulty: str,
    invoke_model,
    feasibility_check,
    original_evidence: dict[str, str],
    overlap_threshold: float,
    existing_specs: list[SpecRecord],
    max_attempts: int,
) -> SpecRecord | None:
    try:
        return _generate_with_regeneration(
            weakness,
            diversity_tags=diversity_tags,
            spec_index=spec_index,
            difficulty=difficulty,
            invoke_model=invoke_model,
            feasibility_check=feasibility_check,
            original_evidence=original_evidence,
            overlap_threshold=overlap_threshold,
            existing_specs=existing_specs,
            max_attempts=max_attempts,
        )
    except Exception as error:
        try:
            fallback_spec = _fallback_spec_for_weakness(
                weakness,
                diversity_tags=diversity_tags,
                spec_index=spec_index,
                difficulty=difficulty,
                original_evidence=original_evidence,
                must_avoid_constraints=[f"avoid duplicates of {spec.spec_id}" for spec in existing_specs],
            )
        except Exception:
            fallback_spec = None
        if fallback_spec is not None:
            feasibility_result = check_feasibility(
                fallback_spec,
                original_evidence=original_evidence,
                feasibility_check=feasibility_check,
            )
            if feasibility_result.accepted:
                append_jsonl(
                    output_path.parent / "errors.jsonl",
                    [
                        {
                            "stage": "synthesize",
                            "weakness": weakness_key(weakness),
                            "event_type": "fallback_used",
                            "message": "deterministic fallback spec accepted",
                        }
                    ],
                )
                return fallback_spec
        append_jsonl(
            output_path.parent / "errors.jsonl",
            [
                {
                    "stage": "synthesize",
                    "weakness": weakness_key(weakness),
                    "error_type": "spec_generation_failed",
                    "message": str(error),
                }
            ],
        )
        return None


def _generate_with_regeneration(
    weakness: WeaknessEntry,
    *,
    diversity_tags,
    spec_index: int,
    difficulty: str,
    invoke_model,
    feasibility_check,
    original_evidence: dict[str, str],
    overlap_threshold: float,
    existing_specs: list[SpecRecord],
    max_attempts: int,
) -> SpecRecord:
    must_avoid_constraints = _dedupe_constraints(
        [f"avoid duplicates of {spec.spec_id}" for spec in existing_specs]
    )
    repair_context: dict[str, str] = {"mode": "initial_generation", "reason": ""}

    for _ in range(max_attempts):
        spec = generate_spec(
            weakness,
            diversity_tags=diversity_tags,
            invoke_model=invoke_model,
            original_evidence=original_evidence,
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
            repair_context=repair_context,
        )
        diversity_result = assign_diversity_tags(
            _relevant_existing_specs(existing_specs, weakness),
            spec.diversity_tags,
            overlap_threshold,
        )
        feasibility_result = check_feasibility(
            spec,
            original_evidence=original_evidence,
            feasibility_check=feasibility_check,
        )
        if diversity_result.accepted and feasibility_result.accepted:
            return spec
        if not diversity_result.accepted:
            repair_context = {
                "mode": _repair_mode_for_diversity_reason(diversity_result.reason),
                "reason": diversity_result.reason,
            }
            must_avoid_constraints = _dedupe_constraints(
                must_avoid_constraints + [f"repair diversity issue: {diversity_result.reason}"]
            )
        if not feasibility_result.accepted:
            repair_context = {
                "mode": _repair_mode_for_feasibility_reason(feasibility_result.reason),
                "reason": feasibility_result.reason,
                "missing_contracts": list(feasibility_result.missing_contracts),
            }
            must_avoid_constraints = _dedupe_constraints(
                must_avoid_constraints + [f"repair feasibility issue: {feasibility_result.reason}"]
            )

    raise ValueError(f"Failed to synthesize a feasible spec for weakness {weakness_key(weakness)}")


def _fallback_spec_for_weakness(
    weakness: WeaknessEntry,
    *,
    diversity_tags,
    spec_index: int,
    difficulty: str,
    original_evidence: dict[str, str],
    must_avoid_constraints: list[str],
) -> SpecRecord | None:
    key = weakness_key(weakness)
    evidence_reference = _evidence_reference(original_evidence)
    fallback_language = _fallback_language_profile(original_evidence)
    if key == "function_name_mismatch":
        wrong_entrypoint, target_entrypoint = _entrypoints_from_evidence(original_evidence)
        return generate_spec(
            weakness,
            diversity_tags=diversity_tags,
            invoke_model=lambda payload: {
                "algorithm_type": "simulation",
                "difficulty": difficulty,
                "narrative_theme": diversity_tags.narrative_theme,
                "constraints": {
                    "n_range": [1, 200],
                    "value_range": [0, 1000],
                    "time_limit": "1s",
                    "memory_limit": "256MB",
                },
                "key_trap": (
                    f"The trap reproduces the original wrong callable `{wrong_entrypoint}` "
                    f"instead of the exact `{target_entrypoint}` entry point required by the harness."
                ),
                "must_cover": [
                    f"exact callable entry point {target_entrypoint}",
                    "single exact public function contract",
                ],
                "must_avoid": [
                    "alternate public function names",
                    f"renaming the public entry point to {wrong_entrypoint} or other helper wrappers",
                ],
                "verification_spec": {
                    "min_test_cases": 4,
                    "must_include_edge_cases": ["single value input"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(n^2)",
                },
                "generation_hints": {
                    "solution_approach": f"Implement the exact {target_entrypoint} entry point expected by the harness.",
                    "common_wrong_approach": f"Expose {wrong_entrypoint} instead of {target_entrypoint}.",
                    "distinguishing_test": f"Call {target_entrypoint} directly and reject any alternate public function name.",
                },
                "language_constraint": fallback_language,
            },
            original_evidence=original_evidence,
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
            repair_context={"mode": "fallback_generation", "reason": "deterministic function name mismatch fallback"},
        )
    if key == "markdown_formatting":
        executable_noun = _executable_code_noun(fallback_language["target_languages"][0])
        return generate_spec(
            weakness,
            diversity_tags=diversity_tags,
            invoke_model=lambda payload: {
                "algorithm_type": "simulation",
                "difficulty": difficulty,
                "narrative_theme": diversity_tags.narrative_theme,
                "constraints": {
                    "n_range": [1, 200],
                    "value_range": [0, 1000],
                    "time_limit": "1s",
                    "memory_limit": "256MB",
                },
                "key_trap": (
                    f"The trap reproduces the original evidence `{evidence_reference}` "
                    f"with markdown fences instead of raw executable {executable_noun}."
                ),
                "must_cover": [f"raw executable {executable_noun} output"],
                "must_avoid": ["fenced code blocks", "wrapping delimiters"],
                "verification_spec": {
                    "min_test_cases": 4,
                    "must_include_edge_cases": ["single value input"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(n^2)",
                },
                "generation_hints": {
                    "solution_approach": f"Return raw executable {executable_noun} only.",
                    "common_wrong_approach": "Wrap the answer in markdown fences.",
                    "distinguishing_test": "Reject output if it contains ``` or stray backticks.",
                },
                "language_constraint": fallback_language,
            },
            original_evidence=original_evidence,
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
            repair_context={"mode": "fallback_generation", "reason": "deterministic markdown formatting fallback"},
        )
    if key == "missing_code_block":
        _, target_entrypoint = _entrypoints_from_evidence(original_evidence)
        entrypoint_clause = _missing_code_block_entrypoint_clause(fallback_language["target_languages"][0], target_entrypoint)
        executable_clause = _missing_code_block_executable_clause(fallback_language["target_languages"][0], entrypoint_clause)
        return generate_spec(
            weakness,
            diversity_tags=diversity_tags,
            invoke_model=lambda payload: {
                "algorithm_type": "simulation",
                "difficulty": difficulty,
                "narrative_theme": diversity_tags.narrative_theme,
                "constraints": {
                    "n_range": [1, 200],
                    "value_range": [0, 1000],
                    "time_limit": "1s",
                    "memory_limit": "256MB",
                },
                "key_trap": (
                    "The trap fails when the solver explains the approach instead of returning "
                    f"{entrypoint_clause}."
                ),
                "must_cover": [f"{executable_clause}"],
                "must_avoid": ["explanation-only answers", "prose-only final responses"],
                "verification_spec": {
                    "min_test_cases": 4,
                    "must_include_edge_cases": ["single value input"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(n^2)",
                },
                "generation_hints": {
                    "solution_approach": _missing_code_block_solution_approach(
                        fallback_language["target_languages"][0],
                        entrypoint_clause,
                    ),
                    "common_wrong_approach": "Explain the intended code without emitting it.",
                    "distinguishing_test": f"Check that {entrypoint_clause} exists as runnable code.",
                },
                "language_constraint": fallback_language,
            },
            original_evidence=original_evidence,
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
            repair_context={"mode": "fallback_generation", "reason": "deterministic missing code fallback"},
        )
    if key == "syntax_error":
        executable_clause = _syntax_error_executable_clause(fallback_language["target_languages"][0])
        wrong_fragment = evidence_reference or "the original incomplete code fragment"
        syntax_avoid = _syntax_error_must_avoid(fallback_language["target_languages"][0])
        spec = generate_spec(
            weakness,
            diversity_tags=diversity_tags,
            invoke_model=lambda payload: {
                "algorithm_type": "simulation",
                "difficulty": difficulty,
                "narrative_theme": diversity_tags.narrative_theme,
                "constraints": {
                    "n_range": [1, 200],
                    "value_range": [0, 1000],
                    "time_limit": "1s",
                    "memory_limit": "256MB",
                },
                "key_trap": f"The trap reproduces the original incomplete syntax fragment `{wrong_fragment}` instead of executable code that parses cleanly.",
                "must_cover": [executable_clause],
                "must_avoid": syntax_avoid,
                "verification_spec": {
                    "min_test_cases": 4,
                    "must_include_edge_cases": ["single value input"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(n^2)",
                },
                "generation_hints": {
                    "solution_approach": _syntax_error_solution_approach(fallback_language["target_languages"][0]),
                    "common_wrong_approach": f"Repeat the incomplete syntax from `{wrong_fragment}`.",
                    "distinguishing_test": _syntax_error_distinguishing_test(fallback_language["target_languages"][0]),
                },
                "language_constraint": fallback_language,
            },
            original_evidence=original_evidence,
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
            repair_context={"mode": "fallback_generation", "reason": "deterministic syntax error fallback"},
        )
        return _rewrite_syntax_error_fallback_spec(
            spec,
            fallback_language["target_languages"][0],
            syntax_avoid,
            must_avoid_constraints,
        )
    if key == "non_executable_code":
        return generate_spec(
            weakness,
            diversity_tags=diversity_tags,
            invoke_model=lambda payload: {
                "algorithm_type": "simulation",
                "difficulty": difficulty,
                "narrative_theme": diversity_tags.narrative_theme,
                "constraints": {
                    "n_range": [1, 200],
                    "value_range": [0, 1000],
                    "time_limit": "1s",
                    "memory_limit": "256MB",
                },
                "key_trap": "The trap reproduces the original translation or prompt echo instead of runnable code that defines the requested function.",
                "must_cover": ["runnable code output with the requested executable implementation present"],
                "must_avoid": [
                    "explanation-only responses",
                    "translation-only output",
                    "prompt echo instead of executable code",
                    "prose-only final responses",
                ],
                "verification_spec": {
                    "min_test_cases": 4,
                    "must_include_edge_cases": ["single value input"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(n^2)",
                },
                "generation_hints": {
                    "solution_approach": "Return runnable code that defines the requested function or entry point.",
                    "common_wrong_approach": "Echo, translate, or explain the prompt without emitting executable code.",
                    "distinguishing_test": "Reject outputs that cannot be imported or executed by the harness.",
                },
                "language_constraint": fallback_language,
            },
            original_evidence=original_evidence,
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
            repair_context={"mode": "fallback_generation", "reason": "deterministic non-executable code fallback"},
        )
    return None


def _evidence_reference(original_evidence: dict[str, str]) -> str:
    wrong_line = original_evidence.get("wrong_line", "")
    for line in wrong_line.splitlines():
        if line.strip().startswith("```"):
            continue
        candidate = line.strip("` ").strip()
        if candidate:
            return candidate[:120]
    return wrong_line[:120] or "original failing output"


def _fallback_language_profile(original_evidence: dict[str, str]) -> dict[str, object]:
    profile = infer_language_profile(original_evidence)
    if profile.primary_language == "unknown":
        return {
            "target_languages": ["python"],
            "language_specific": False,
        }
    return {
        "target_languages": list(profile.target_languages),
        "language_specific": profile.language_specific,
    }


def _dedupe_constraints(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _executable_code_noun(language: str) -> str:
    if language == "r":
        return "R code"
    if language == "java":
        return "Java code"
    if language == "cpp":
        return "C++ code"
    if language == "javascript":
        return "JavaScript code"
    if language == "typescript":
        return "TypeScript code"
    if language == "go":
        return "Go code"
    if language == "rust":
        return "Rust code"
    return "code"


def _missing_code_block_entrypoint_clause(language: str, target_entrypoint: str) -> str:
    if target_entrypoint == "the requested entry point":
        return "the requested executable implementation"
    if language == "java":
        return f"the requested `{target_entrypoint}` method"
    return f"the requested `{target_entrypoint}` entry point"


def _missing_code_block_executable_clause(language: str, entrypoint_clause: str) -> str:
    if language == "java":
        return f"executable Java code output with {entrypoint_clause}"
    return f"executable code output with {entrypoint_clause}"


def _missing_code_block_solution_approach(language: str, entrypoint_clause: str) -> str:
    if language == "java":
        return f"Return {entrypoint_clause} directly as executable Java code."
    return f"Return {entrypoint_clause} directly as executable code."


def _syntax_error_executable_clause(language: str) -> str:
    if language == "java":
        return "syntactically complete executable Java code that compiles cleanly"
    return "syntactically complete executable code"


def _syntax_error_solution_approach(language: str) -> str:
    if language == "java":
        return "Return syntactically valid executable Java code."
    return "Return syntactically valid executable code."


def _syntax_error_distinguishing_test(language: str) -> str:
    if language == "java":
        return "Compile the final Java code before execution."
    return "Parse the final code before execution."


def _syntax_error_must_avoid(language: str) -> list[str]:
    if language == "java":
        return ["incomplete code", "missing braces", "malformed method headers"]
    if language in {"cpp", "javascript", "typescript", "go", "rust", "r"}:
        return ["incomplete code", "missing required punctuation", "malformed code structure"]
    return ["incomplete code", "missing colons", "malformed function headers"]


def _rewrite_syntax_error_fallback_spec(
    spec: SpecRecord,
    language: str,
    syntax_avoid: list[str],
    must_avoid_constraints: list[str],
) -> SpecRecord:
    if language == "python":
        return spec
    return spec.model_copy(
        update={
            "problem_spec": spec.problem_spec.model_copy(
                update={"must_avoid": syntax_avoid + must_avoid_constraints}
            )
        }
    )


def _entrypoints_from_evidence(original_evidence: dict[str, str]) -> tuple[str, str]:
    wrong_names = _function_names_in_text(original_evidence.get("wrong_line", ""))
    correct_names = _function_names_in_text(original_evidence.get("correct_approach", ""))
    failed_names = _function_names_in_text(original_evidence.get("failed_test", ""))
    wrong = wrong_names[0] if wrong_names else "the wrong callable"
    target = (correct_names or failed_names or wrong_names or ["the requested entry point"])[-1]
    return wrong, target


def _function_names_in_text(text: str) -> list[str]:
    import re

    names: list[str] = []
    patterns = [
        r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bfunction\s+named\s+['`\"]?([A-Za-z_][A-Za-z0-9_]*)",
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


def _repair_mode_for_feasibility_reason(reason: str) -> str:
    lowered = reason.lower()
    if "exact public entry-point" in lowered or "alternate public function names" in lowered:
        return "contract_mismatch"
    if "raw executable output" in lowered or "markdown fences" in lowered or "wrapping delimiters" in lowered:
        return "raw_output_required"
    if "syntactically complete executable code" in lowered or "incomplete code forms" in lowered:
        return "syntax_completion_required"
    if "executable code output" in lowered or "explanation-only" in lowered or "prose" in lowered:
        return "executable_code_required"
    return "feasibility_retry"


def _repair_mode_for_diversity_reason(reason: str) -> str:
    lowered = reason.lower()
    if "similar" in lowered or "duplicate" in lowered or "overlap" in lowered:
        return "duplicate_diversity_pattern"
    return "diversity_retry"


def read_weakness_report(path: Path) -> WeaknessReport:
    return WeaknessReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


def read_specs(path: Path) -> list[SpecRecord]:
    if not path.exists():
        return []
    return [SpecRecord.model_validate(row) for row in read_jsonl(path)]


def read_diagnoses(path: Path) -> list[DiagnosisRecord]:
    if not path.exists():
        return []
    return [DiagnosisRecord.model_validate(row) for row in read_jsonl(path)]


def build_original_evidence_map(
    report: WeaknessReport,
    diagnoses: list[DiagnosisRecord],
) -> dict[str, dict[str, str]]:
    sample_evidence_map = build_sample_evidence_map(report, diagnoses)
    evidence_map: dict[str, dict[str, str]] = {}
    for key, sample_evidence in sample_evidence_map.items():
        evidence_map[key] = _merge_evidence_dicts(sample_evidence[:3])
    return evidence_map


def build_sample_evidence_map(
    report: WeaknessReport,
    diagnoses: list[DiagnosisRecord],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[DiagnosisRecord]] = {}
    for diagnosis in diagnoses:
        for sub_tag in diagnosis.sub_tags:
            canonical_tag = report.tag_mappings.get(sub_tag, sub_tag)
            grouped.setdefault(canonical_tag, []).append(diagnosis)

    evidence_map: dict[str, list[dict[str, str]]] = {}
    for weakness in sorted(report.weaknesses, key=lambda item: item.rank):
        key = weakness_key(weakness)
        if key in evidence_map:
            continue
        sample_task_id_set = set(weakness.sample_task_ids)
        matches = [
            diagnosis
            for diagnosis in diagnoses
            if diagnosis.task_id in sample_task_id_set
        ]
        if weakness.sample_task_ids:
            order = {task_id: index for index, task_id in enumerate(weakness.sample_task_ids)}
            matches.sort(key=lambda diagnosis: order.get(diagnosis.task_id, len(order)))
        if not matches:
            matches = grouped.get(key, [])
        if matches:
            evidence_map[key] = [_diagnosis_evidence(diagnosis) for diagnosis in matches[:3]]

    return evidence_map


def _difficulty_for_slot(slot_index: int, count: int, distribution: str) -> str:
    if distribution == "weighted_hard":
        return "medium" if slot_index % 3 == 1 else "hard"
    return "medium" if slot_index < count / 2 else "hard"


def _evidence_for_weakness(
    weakness: WeaknessEntry,
    original_evidence: dict[str, dict[str, str]] | None,
) -> dict[str, str]:
    key = weakness_key(weakness)
    if original_evidence and key in original_evidence:
        return original_evidence[key]

    return {
        "wrong_line": f"Representative {key} failure pattern.",
        "correct_approach": f"Implement the required {key} contract correctly.",
        "failed_test": f"Representative failure unavailable for {key}",
    }


def _evidence_for_slot(
    weakness: WeaknessEntry,
    original_evidence: dict[str, dict[str, str]] | None,
    sample_evidence_map: dict[str, list[dict[str, str]]] | None,
    slot_index: int,
) -> dict[str, str]:
    key = weakness_key(weakness)
    if sample_evidence_map and key in sample_evidence_map:
        sample_evidence = sample_evidence_map[key]
        if sample_evidence:
            return sample_evidence[slot_index % len(sample_evidence)]
    return _evidence_for_weakness(weakness, original_evidence)


def _relevant_existing_specs(existing_specs: list[SpecRecord], weakness: WeaknessEntry) -> list[SpecRecord]:
    target_key = weakness_key(weakness)
    return [
        spec
        for spec in existing_specs
        if weakness_key(spec.target_weakness) == target_key
    ]


def _merge_evidence(diagnoses: list[DiagnosisRecord]) -> dict[str, str]:
    return _merge_evidence_dicts([_diagnosis_evidence(diagnosis) for diagnosis in diagnoses])


def _diagnosis_evidence(diagnosis: DiagnosisRecord) -> dict[str, str]:
    return {
        "wrong_line": diagnosis.evidence.wrong_line,
        "correct_approach": diagnosis.evidence.correct_approach,
        "failed_test": diagnosis.evidence.failed_test,
    }


def _merge_evidence_dicts(evidence_rows: list[dict[str, str]]) -> dict[str, str]:
    wrong_lines: list[str] = []
    correct_approaches: list[str] = []
    failed_tests: list[str] = []

    for evidence in evidence_rows:
        if evidence["wrong_line"] not in wrong_lines:
            wrong_lines.append(evidence["wrong_line"])
        if evidence["correct_approach"] not in correct_approaches:
            correct_approaches.append(evidence["correct_approach"])
        if evidence["failed_test"] not in failed_tests:
            failed_tests.append(evidence["failed_test"])

    return {
        "wrong_line": " | ".join(wrong_lines),
        "correct_approach": " | ".join(correct_approaches),
        "failed_test": " | ".join(failed_tests),
    }
