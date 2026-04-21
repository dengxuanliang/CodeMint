from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codemint.models.spec import SpecRecord
from codemint.prompts.registry import load_prompt
from codemint.synthesize.contracts import extract_contract_signals
from codemint.synthesize.generate import has_concrete_evidence_reference


@dataclass(frozen=True, slots=True)
class FeasibilityResult:
    accepted: bool
    reason: str
    missing_contracts: list[str] = field(default_factory=list)


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
            missing_contracts=_missing_contracts_for_spec(spec),
        )
    if not _passes_function_name_mismatch_checks(spec):
        return FeasibilityResult(
            accepted=False,
            reason="Function-name mismatch weakness spec must require a single exact public entry-point contract and forbid alternate public function names.",
            missing_contracts=_missing_contracts_for_spec(spec),
        )
    if not _passes_markdown_formatting_checks(spec):
        return FeasibilityResult(
            accepted=False,
            reason="Markdown-formatting weakness spec must require raw executable output and forbid markdown fences or wrapping delimiters.",
            missing_contracts=_missing_contracts_for_spec(spec),
        )
    if not _passes_syntax_error_checks(spec):
        return FeasibilityResult(
            accepted=False,
            reason="Syntax-error weakness spec must require syntactically complete executable code and forbid incomplete code forms.",
            missing_contracts=_missing_contracts_for_spec(spec),
        )
    if not _passes_non_executable_code_checks(spec):
        return FeasibilityResult(
            accepted=False,
            reason="Non-executable-code weakness spec must require executable output presence and forbid explanation-only or non-runnable responses.",
            missing_contracts=_missing_contracts_for_spec(spec),
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
        missing_contracts = [str(item) for item in response.get("missing_contracts", [])]
        return FeasibilityResult(accepted=accepted, reason=reason, missing_contracts=missing_contracts)
    raise TypeError("Unsupported feasibility response type")


def _missing_contracts_for_spec(spec: SpecRecord) -> list[str]:
    signals = extract_contract_signals(spec)
    missing: list[str] = []

    if "function_name_mismatch" in spec.target_weakness.sub_tags:
        if not signals.requires_exact_public_entry_point:
            missing.append("requires_exact_public_entry_point")
        if not signals.forbids_alternate_public_names:
            missing.append("forbids_alternate_public_names")

    if "missing_code_block" in spec.target_weakness.sub_tags:
        if not signals.requires_executable_code_output:
            missing.append("requires_executable_code_output")
        if not signals.forbids_explanation_only:
            missing.append("forbids_explanation_only")

    if "markdown_formatting" in spec.target_weakness.sub_tags:
        if not signals.requires_raw_executable_output:
            missing.append("requires_raw_executable_output")
        if not signals.forbids_markdown_wrapping:
            missing.append("forbids_markdown_wrapping")

    if "syntax_error" in spec.target_weakness.sub_tags:
        if not signals.requires_syntactic_completeness:
            missing.append("requires_syntactic_completeness")
        if not signals.forbids_incomplete_code:
            missing.append("forbids_incomplete_code")

    if "non_executable_code" in spec.target_weakness.sub_tags:
        if not signals.requires_executable_code_output:
            missing.append("requires_executable_code_output")
        if not signals.forbids_explanation_only:
            missing.append("forbids_explanation_only")

    return missing


def _passes_missing_code_block_checks(spec: SpecRecord) -> bool:
    if "missing_code_block" not in spec.target_weakness.sub_tags:
        return True

    signals = extract_contract_signals(spec)
    return signals.requires_executable_code_output and signals.forbids_explanation_only


def _passes_function_name_mismatch_checks(spec: SpecRecord) -> bool:
    if "function_name_mismatch" not in spec.target_weakness.sub_tags:
        return True

    signals = extract_contract_signals(spec)
    return signals.requires_exact_public_entry_point and signals.forbids_alternate_public_names


def _passes_markdown_formatting_checks(spec: SpecRecord) -> bool:
    if "markdown_formatting" not in spec.target_weakness.sub_tags:
        return True

    signals = extract_contract_signals(spec)
    return signals.requires_raw_executable_output and signals.forbids_markdown_wrapping


def _passes_syntax_error_checks(spec: SpecRecord) -> bool:
    if "syntax_error" not in spec.target_weakness.sub_tags:
        return True

    signals = extract_contract_signals(spec)
    return signals.requires_syntactic_completeness and signals.forbids_incomplete_code


def _passes_non_executable_code_checks(spec: SpecRecord) -> bool:
    if not any(tag in {"missing_code_block", "non_executable_code"} for tag in spec.target_weakness.sub_tags):
        return True

    signals = extract_contract_signals(spec)
    return signals.requires_executable_code_output and signals.forbids_explanation_only
