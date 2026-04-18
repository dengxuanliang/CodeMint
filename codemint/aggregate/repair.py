from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

from codemint.models.diagnosis import DiagnosisRecord


VerificationLevel = Literal["auto", "exec_api", "cross_model", "self_check"]
VerificationStatus = Literal["passed", "failed", "unverified"]
ResolvedVerificationLevel = Literal["exec_api", "cross_model", "self_check"]
VerifierResult: TypeAlias = str | None


class VerificationResultDict(TypedDict):
    level: ResolvedVerificationLevel
    status: Literal["passed", "failed"]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    level: ResolvedVerificationLevel
    status: Literal["passed", "failed"]


def verify_repair(
    diagnosis: DiagnosisRecord,
    *,
    verification_level: VerificationLevel = "auto",
    exec_api_reachable: Callable[[], bool] | None = None,
    exec_api_verifier: Callable[[DiagnosisRecord], VerifierResult] | None = None,
    cross_model_verifier: Callable[[DiagnosisRecord], VerifierResult] | None = None,
    self_check_verifier: Callable[[DiagnosisRecord], VerifierResult] | None = None,
) -> VerificationResult:
    levels = _verification_levels(
        verification_level=verification_level,
        exec_api_reachable=exec_api_reachable,
    )
    for level in levels:
        verifier = _select_verifier(
            level,
            exec_api_verifier=exec_api_verifier,
            cross_model_verifier=cross_model_verifier,
            self_check_verifier=self_check_verifier,
        )
        status = verifier(diagnosis)
        if status is not None:
            return VerificationResult(level=level, status=status)

    raise ValueError("No verification path produced a result")


def repair_diagnosis(
    diagnosis: DiagnosisRecord,
    *,
    verification_level: VerificationLevel = "auto",
    verify: Callable[[DiagnosisRecord, VerificationLevel], VerificationResultDict | VerificationResult]
    | None = None,
    rediagnose: Callable[[DiagnosisRecord], DiagnosisRecord],
) -> DiagnosisRecord:
    current = diagnosis
    retry_count = 0
    verifier = verify or default_verify

    while True:
        verification = _coerce_verification_result(verifier(current, verification_level))
        if verification.status != "failed":
            return _apply_verification_metadata(
                current,
                verification_level=verification.level,
                verification_status="passed",
            )

        if retry_count == 0:
            current = rediagnose(current)
            retry_count += 1
            continue

        current = current.model_copy(deep=True)
        current.confidence = min(current.confidence, 0.5)
        return _apply_verification_metadata(
            current,
            verification_level=verification.level,
            verification_status="unverified",
        )


def _verification_levels(
    *,
    verification_level: VerificationLevel,
    exec_api_reachable: Callable[[], bool] | None,
) -> list[ResolvedVerificationLevel]:
    if verification_level == "auto":
        if exec_api_reachable and exec_api_reachable():
            return ["exec_api", "cross_model", "self_check"]
        return ["cross_model", "self_check"]
    if verification_level == "exec_api":
        return ["exec_api", "cross_model", "self_check"]
    if verification_level == "cross_model":
        return ["cross_model", "self_check"]
    return ["self_check"]


def _select_verifier(
    level: ResolvedVerificationLevel,
    *,
    exec_api_verifier: Callable[[DiagnosisRecord], VerifierResult] | None,
    cross_model_verifier: Callable[[DiagnosisRecord], VerifierResult] | None,
    self_check_verifier: Callable[[DiagnosisRecord], VerifierResult] | None,
) -> Callable[[DiagnosisRecord], VerifierResult]:
    if level == "exec_api":
        return exec_api_verifier or _default_unavailable_verifier
    if level == "cross_model":
        return cross_model_verifier or _default_unavailable_verifier
    return self_check_verifier or _default_passing_verifier


def default_verify(
    diagnosis: DiagnosisRecord,
    verification_level: VerificationLevel,
) -> VerificationResult:
    return verify_repair(diagnosis, verification_level=verification_level)


def _default_unavailable_verifier(_: DiagnosisRecord) -> None:
    return None


def _default_passing_verifier(_: DiagnosisRecord) -> str:
    return "passed"


def _apply_verification_metadata(
    diagnosis: DiagnosisRecord,
    *,
    verification_level: ResolvedVerificationLevel,
    verification_status: VerificationStatus,
) -> DiagnosisRecord:
    updated = diagnosis.model_copy(deep=True)
    updated.enriched_labels["verification_level"] = verification_level
    updated.enriched_labels["verification_status"] = verification_status
    return updated


def _coerce_verification_result(
    value: VerificationResultDict | VerificationResult,
) -> VerificationResult:
    if isinstance(value, VerificationResult):
        return value
    return VerificationResult(level=value["level"], status=value["status"])
