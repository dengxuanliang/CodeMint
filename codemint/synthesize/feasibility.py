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
