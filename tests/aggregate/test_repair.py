from __future__ import annotations

from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord


def test_verification_level_auto_prefers_exec_api_when_reachable() -> None:
    from codemint.aggregate.repair import verify_repair

    diagnosis = _diagnosis(1)
    calls: list[str] = []

    def exec_probe() -> bool:
        calls.append("probe")
        return True

    def exec_verifier(record: DiagnosisRecord) -> str:
        calls.append(f"exec:{record.task_id}")
        return "passed"

    def cross_verifier(record: DiagnosisRecord) -> str:
        calls.append(f"cross:{record.task_id}")
        return "failed"

    def self_verifier(record: DiagnosisRecord) -> str:
        calls.append(f"self:{record.task_id}")
        return "failed"

    result = verify_repair(
        diagnosis,
        verification_level="auto",
        exec_api_reachable=exec_probe,
        exec_api_verifier=exec_verifier,
        cross_model_verifier=cross_verifier,
        self_check_verifier=self_verifier,
    )

    assert result.level == "exec_api"
    assert result.status == "passed"
    assert calls == ["probe", "exec:1"]


def test_second_verification_failure_forces_unverified_and_caps_confidence() -> None:
    from codemint.aggregate.repair import repair_diagnosis

    calls: list[str] = []

    def verifier(record: DiagnosisRecord, level: str):
        calls.append(f"verify:{record.task_id}:{level}")
        if len(calls) == 1:
            return {"level": "self_check", "status": "failed"}
        return {"level": "cross_model", "status": "failed"}

    def rediagnose(record: DiagnosisRecord) -> DiagnosisRecord:
        calls.append(f"rediagnose:{record.task_id}")
        updated = record.model_copy(deep=True)
        updated.description = "Retried diagnosis"
        updated.confidence = 0.92
        return updated

    repaired = repair_diagnosis(
        _diagnosis(7, confidence=0.91),
        verification_level="self_check",
        verify=verifier,
        rediagnose=rediagnose,
    )

    assert repaired.description == "Retried diagnosis"
    assert repaired.confidence == 0.5
    assert repaired.enriched_labels["verification_status"] == "unverified"
    assert repaired.enriched_labels["verification_level"] == "cross_model"
    assert calls == [
        "verify:7:self_check",
        "rediagnose:7",
        "verify:7:self_check",
    ]


def _diagnosis(task_id: int, *, confidence: float = 0.9) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task_id,
        fault_type="implementation",
        sub_tags=["off_by_one", "loop_bound"],
        severity="medium",
        description="Diagnosis",
        evidence=DiagnosisEvidence(
            wrong_line="for i in range(n + 1):",
            correct_approach="Use the correct upper bound.",
            failed_test="assert solve([1, 2]) == 3",
        ),
        enriched_labels={},
        confidence=confidence,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )
