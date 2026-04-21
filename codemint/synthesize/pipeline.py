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
                original_evidence=_evidence_for_weakness(weakness, evidence_map),
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
        fallback_spec = _fallback_spec_for_weakness(
            weakness,
            diversity_tags=diversity_tags,
            spec_index=spec_index,
            difficulty=difficulty,
            original_evidence=original_evidence,
            must_avoid_constraints=[f"avoid duplicates of {spec.spec_id}" for spec in existing_specs],
        )
        if fallback_spec is not None:
            feasibility_result = check_feasibility(
                fallback_spec,
                original_evidence=original_evidence,
                feasibility_check=feasibility_check,
            )
            if feasibility_result.accepted:
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
    must_avoid_constraints = [f"avoid duplicates of {spec.spec_id}" for spec in existing_specs]
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
        diversity_result = assign_diversity_tags(existing_specs, spec.diversity_tags, overlap_threshold)
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
            must_avoid_constraints.append(f"repair diversity issue: {diversity_result.reason}")
        if not feasibility_result.accepted:
            repair_context = {
                "mode": _repair_mode_for_feasibility_reason(feasibility_result.reason),
                "reason": feasibility_result.reason,
                "missing_contracts": list(feasibility_result.missing_contracts),
            }
            must_avoid_constraints.append(f"repair feasibility issue: {feasibility_result.reason}")

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
    if key == "function_name_mismatch":
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
                "key_trap": "The trap reproduces the original solve_value output instead of the exact solve(x) entry point required by the harness.",
                "must_cover": [
                    "exact callable entry point solve(x)",
                    "single exact public function contract",
                ],
                "must_avoid": [
                    "alternate public function names",
                    "renaming the public entry point to solve_value, solver, or helper wrappers",
                ],
                "verification_spec": {
                    "min_test_cases": 4,
                    "must_include_edge_cases": ["single value input"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(n^2)",
                },
                "generation_hints": {
                    "solution_approach": "Implement the exact solve(x) entry point expected by the harness.",
                    "common_wrong_approach": "Expose solve_value(x) or solver(x) instead of solve(x).",
                    "distinguishing_test": "Call solve() directly and reject any alternate public function name.",
                },
                "language_constraint": {
                    "target_languages": ["python"],
                    "language_specific": False,
                },
            },
            original_evidence=original_evidence,
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
            repair_context={"mode": "fallback_generation", "reason": "deterministic function name mismatch fallback"},
        )
    if key == "markdown_formatting":
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
                "key_trap": "The trap reproduces the original ```python fenced solve(x) output instead of raw executable code.",
                "must_cover": ["raw executable code output"],
                "must_avoid": ["fenced code blocks", "wrapping delimiters"],
                "verification_spec": {
                    "min_test_cases": 4,
                    "must_include_edge_cases": ["single value input"],
                    "brute_force_verifiable": True,
                    "brute_force_complexity_limit": "O(n^2)",
                },
                "generation_hints": {
                    "solution_approach": "Return raw executable code only.",
                    "common_wrong_approach": "Wrap the answer in markdown fences.",
                    "distinguishing_test": "Reject output if it contains ``` or stray backticks.",
                },
                "language_constraint": {
                    "target_languages": ["python"],
                    "language_specific": False,
                },
            },
            original_evidence=original_evidence,
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
            repair_context={"mode": "fallback_generation", "reason": "deterministic markdown formatting fallback"},
        )
    if key == "missing_code_block":
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
                "key_trap": "The trap fails when the solver explains the approach instead of returning a callable solve function.",
                "must_cover": ["executable code output with a callable solve function"],
                "must_avoid": ["explanation-only answers", "prose-only final responses"],
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
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
            repair_context={"mode": "fallback_generation", "reason": "deterministic missing code fallback"},
        )
    if key == "syntax_error":
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
                "key_trap": "The trap reproduces the original def solve(x) header without the required colon.",
                "must_cover": ["syntactically complete executable code with a valid solve(x) definition"],
                "must_avoid": ["incomplete code", "missing colons", "malformed function headers"],
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
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
            repair_context={"mode": "fallback_generation", "reason": "deterministic syntax error fallback"},
        )
    return None


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
    grouped: dict[str, list[DiagnosisRecord]] = {}
    for diagnosis in diagnoses:
        for sub_tag in diagnosis.sub_tags:
            canonical_tag = report.tag_mappings.get(sub_tag, sub_tag)
            grouped.setdefault(canonical_tag, []).append(diagnosis)

    evidence_map: dict[str, dict[str, str]] = {}
    for weakness in report.weaknesses:
        key = weakness_key(weakness)
        matches = grouped.get(key, [])
        if not matches and weakness.sample_task_ids:
            matches = [
                diagnosis
                for diagnosis in diagnoses
                if diagnosis.task_id in set(weakness.sample_task_ids)
            ]
        if matches:
            evidence_map[key] = _merge_evidence(matches[:3])

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
        "wrong_line": weakness.collective_diagnosis.refined_root_cause,
        "correct_approach": weakness.collective_diagnosis.capability_cliff,
        "failed_test": f"Reproduce failure around {key}",
    }


def _merge_evidence(diagnoses: list[DiagnosisRecord]) -> dict[str, str]:
    wrong_lines: list[str] = []
    correct_approaches: list[str] = []
    failed_tests: list[str] = []

    for diagnosis in diagnoses:
        if diagnosis.evidence.wrong_line not in wrong_lines:
            wrong_lines.append(diagnosis.evidence.wrong_line)
        if diagnosis.evidence.correct_approach not in correct_approaches:
            correct_approaches.append(diagnosis.evidence.correct_approach)
        if diagnosis.evidence.failed_test not in failed_tests:
            failed_tests.append(diagnosis.evidence.failed_test)

    return {
        "wrong_line": " | ".join(wrong_lines),
        "correct_approach": " | ".join(correct_approaches),
        "failed_test": " | ".join(failed_tests),
    }
