from __future__ import annotations

from typing import Literal

from pydantic import Field

from codemint.models.base import StrictModel


FaultType = Literal["comprehension", "modeling", "implementation", "edge_handling", "surface"]
Severity = Literal["low", "medium", "high"]
DiagnosisSource = Literal["rule_confirmed_by_model", "rule_only", "model_deep", "non_failure"]
SUCCESS_LIKE_SUB_TAGS = frozenset({"correct_output", "correct_execution", "correct_solution", "pass"})


class DiagnosisEvidence(StrictModel):
    wrong_line: str
    correct_approach: str
    failed_test: str


class DiagnosisRecord(StrictModel):
    task_id: int
    fault_type: FaultType
    sub_tags: list[str]
    severity: Severity
    description: str
    evidence: DiagnosisEvidence
    enriched_labels: dict[str, str]
    confidence: float = Field(ge=0.0, le=1.0)
    diagnosis_source: DiagnosisSource
    prompt_version: str

    @property
    def is_failure(self) -> bool:
        normalized_tags = {tag.strip().lower() for tag in self.sub_tags}
        normalized_labels = {
            key.strip().lower(): value.strip().lower()
            for key, value in self.enriched_labels.items()
        }
        failure_like_tags = normalized_tags - SUCCESS_LIKE_SUB_TAGS
        if self.diagnosis_source == "non_failure":
            return False
        if failure_like_tags:
            return True
        if normalized_tags & SUCCESS_LIKE_SUB_TAGS:
            return False
        if normalized_labels.get("status") == "correct_solution":
            return False
        if normalized_labels.get("test_result") == "pass":
            return False
        if normalized_labels.get("functional_correctness") == "true":
            return False
        if self.evidence.wrong_line.strip().upper() == "N/A" and self.evidence.failed_test.strip().upper() == "N/A":
            return False
        return True
