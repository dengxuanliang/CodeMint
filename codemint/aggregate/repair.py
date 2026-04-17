from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

from codemint.models.diagnosis import DiagnosisRecord


VerificationLevel = Literal["auto", "exec_api", "cross_model", "self_check"]
VerificationStatus = Literal["passed", "failed", "unverified"]


class VerificationResultDict(TypedDict):
    level: Literal["exec_api", "cross_model", "self_check"]
    status: Literal["passed", "failed"]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    level: Literal["exec_api", "cross_model", "self_check"]
    status: Literal["passed", "failed"]


def verify_repair(
    diagnosis: DiagnosisRecord,
    *,
    verification_level: VerificationLevel = "auto",
    exec_api_reachable: Callable[[], bool] | None = None,
    exec_api_verifier: Callable[[DiagnosisRecord], str] | None = None,
    cross_model_verifier: Callable[[DiagnosisRecord], str] | None = None,
    self_check_verifier: Callable[[DiagnosisRecord], str] | None = None,
) -> VerificationResult:
    level = _resolve_verification_level(
        verification_level=verification_level,
        exec_api_reachable=exec_api_reachable,
    )
    verifier = _select_verifier(
        level,
        exec_api_verifier=exec_api_verifier,
        cross_model_verifier=cross_model_verifier,
        self_check_verifier=self_check_verifier,
    )
    return VerificationResult(level=level, status=verifier(diagnosis))


def repair_diagnosis(
    diagnosis: DiagnosisRecord,
    *,
    verification_level: VerificationLevel = "auto",
    verify: Callable[[DiagnosisRecord, VerificationLevel], VerificationResultDict | VerificationResult],
    rediagnose: Callable[[DiagnosisRecord], DiagnosisRecord],
) -> DiagnosisRecord:
    current = diagnosis
    retry_count = 0

    while True:
        verification = _coerce_verification_result(verify(current, verification_level))
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


def _resolve_verification_level(
    *,
    verification_level: VerificationLevel,
    exec_api_reachable: Callable[[], bool] | None,
) -> Literal["exec_api", "cross_model", "self_check"]:
    if verification_level == "auto":
        if exec_api_reachable and exec_api_reachable():
            return "exec_api"
        return "cross_model"
    return verification_level


def _select_verifier(
    level: Literal["exec_api", "cross_model", "self_check"],
    *,
    exec_api_verifier: Callable[[DiagnosisRecord], str] | None,
    cross_model_verifier: Callable[[DiagnosisRecord], str] | None,
    self_check_verifier: Callable[[DiagnosisRecord], str] | None,
) -> Callable[[DiagnosisRecord], str]:
    if level == "exec_api":
        return exec_api_verifier or _default_verifier("passed")
    if level == "cross_model":
        return cross_model_verifier or _default_verifier("passed")
    return self_check_verifier or _default_verifier("passed")


def _default_verifier(status: Literal["passed", "failed"]) -> Callable[[DiagnosisRecord], str]:
    def verifier(_: DiagnosisRecord) -> str:
        return status

    return verifier


def _apply_verification_metadata(
    diagnosis: DiagnosisRecord,
    *,
    verification_level: Literal["exec_api", "cross_model", "self_check"],
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
