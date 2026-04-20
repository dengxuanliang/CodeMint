from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codemint.models.spec import SpecRecord
from codemint.prompts.registry import load_prompt
from codemint.synthesize.generate import has_concrete_evidence_reference


@dataclass(frozen=True, slots=True)
class FeasibilityResult:
    accepted: bool
    reason: str


def check_feasibility(
    spec: SpecRecord,
    *,
    original_evidence: dict[str, str],
    feasibility_check=None,
) -> FeasibilityResult:
    prompt = load_prompt("synthesize_feasibility_check")
    if not has_concrete_evidence_reference(spec.problem_spec.key_trap, original_evidence):
        return FeasibilityResult(
            accepted=False,
            reason="Generated key_trap no longer references original evidence.",
        )
    if not _passes_missing_code_block_checks(spec):
        return FeasibilityResult(
            accepted=False,
            reason="Missing-code weakness spec must explicitly require executable code output and forbid explanation-only answers.",
        )
    if not _passes_function_name_mismatch_checks(spec):
        return FeasibilityResult(
            accepted=False,
            reason="Function-name mismatch weakness spec must require a single exact public entry-point contract and forbid alternate public function names.",
        )
    if not _passes_markdown_formatting_checks(spec):
        return FeasibilityResult(
            accepted=False,
            reason="Markdown-formatting weakness spec must require raw executable output and forbid markdown fences or wrapping delimiters.",
        )
    if not _passes_syntax_error_checks(spec):
        return FeasibilityResult(
            accepted=False,
            reason="Syntax-error weakness spec must require syntactically complete executable code and forbid incomplete code forms.",
        )
    if not _passes_non_executable_code_checks(spec):
        return FeasibilityResult(
            accepted=False,
            reason="Non-executable-code weakness spec must require executable output presence and forbid explanation-only or non-runnable responses.",
        )

    if feasibility_check is None:
        return FeasibilityResult(accepted=True, reason=f"{prompt.version}: passed local checks")

    payload = {
        "template": prompt.template,
        "spec": spec.model_dump(mode="json"),
        "original_evidence": original_evidence,
    }
    response = feasibility_check(payload)
    return _normalize_result(response)


def _normalize_result(response: Any) -> FeasibilityResult:
    if isinstance(response, FeasibilityResult):
        return response
    if isinstance(response, bool):
        return FeasibilityResult(
            accepted=response,
            reason="external feasibility check",
        )
    if isinstance(response, dict):
        accepted = bool(response.get("accepted", response.get("feasible", False)))
        reason = str(response.get("reason", "external feasibility check"))
        return FeasibilityResult(accepted=accepted, reason=reason)
    raise TypeError("Unsupported feasibility response type")


def _passes_missing_code_block_checks(spec: SpecRecord) -> bool:
    if "missing_code_block" not in spec.target_weakness.sub_tags:
        return True

    must_cover_text = " ".join(spec.problem_spec.must_cover).lower()
    must_avoid_text = " ".join(spec.problem_spec.must_avoid).lower()
    requires_code_output = ("executable code" in must_cover_text) or ("callable solve function" in must_cover_text)
    forbids_explanation = ("explanation-only" in must_avoid_text) or ("prose" in must_avoid_text)
    return requires_code_output and forbids_explanation


def _passes_function_name_mismatch_checks(spec: SpecRecord) -> bool:
    if "function_name_mismatch" not in spec.target_weakness.sub_tags:
        return True

    must_cover_text = " ".join(spec.problem_spec.must_cover).lower()
    must_avoid_text = " ".join(spec.problem_spec.must_avoid).lower()
    requires_exact_entry = ("exact callable entry point" in must_cover_text) or ("solve(x)" in must_cover_text)
    forbids_alternates = ("alternate public function names" in must_avoid_text) or ("rename" in must_avoid_text)
    return requires_exact_entry and forbids_alternates


def _passes_markdown_formatting_checks(spec: SpecRecord) -> bool:
    if "markdown_formatting" not in spec.target_weakness.sub_tags:
        return True

    must_cover_text = " ".join(spec.problem_spec.must_cover).lower()
    must_avoid_text = " ".join(spec.problem_spec.must_avoid).lower()
    requires_raw_output = ("raw executable code" in must_cover_text) or ("raw solve" in must_cover_text)
    forbids_fences = ("fenced code blocks" in must_avoid_text) or ("markdown fences" in must_avoid_text) or (
        "wrapping delimiters" in must_avoid_text
    )
    return requires_raw_output and forbids_fences


def _passes_syntax_error_checks(spec: SpecRecord) -> bool:
    if "syntax_error" not in spec.target_weakness.sub_tags:
        return True

    must_cover_text = " ".join(spec.problem_spec.must_cover).lower()
    must_avoid_text = " ".join(spec.problem_spec.must_avoid).lower()
    requires_syntactic_completeness = ("syntactically complete executable code" in must_cover_text) or (
        "valid solve(x) definition" in must_cover_text
    )
    forbids_incomplete_code = ("incomplete code" in must_avoid_text) or ("missing colons" in must_avoid_text) or (
        "missing bodies" in must_avoid_text
    )
    return requires_syntactic_completeness and forbids_incomplete_code


def _passes_non_executable_code_checks(spec: SpecRecord) -> bool:
    if not any(tag in {"missing_code_block", "non_executable_code"} for tag in spec.target_weakness.sub_tags):
        return True

    must_cover_text = " ".join(spec.problem_spec.must_cover).lower()
    must_avoid_text = " ".join(spec.problem_spec.must_avoid).lower()
    requires_output_presence = ("executable output presence" in must_cover_text) or ("executable code output" in must_cover_text)
    forbids_non_runnable = ("explanation-only" in must_avoid_text) or ("non-runnable" in must_avoid_text) or ("prose" in must_avoid_text)
    return requires_output_presence and forbids_non_runnable
