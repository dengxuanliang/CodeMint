from __future__ import annotations

from typing import Literal

from pydantic import Field

from codemint.models.base import StrictModel


FaultType = Literal["comprehension", "modeling", "implementation", "edge_handling", "surface"]
Severity = Literal["low", "medium", "high"]
DiagnosisSource = Literal["rule_confirmed_by_model", "rule_only", "model_deep"]


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
