from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


FaultType = Literal["comprehension", "modeling", "implementation", "edge_handling", "surface"]
Severity = Literal["low", "medium", "high"]
DiagnosisSource = Literal["rule_confirmed_by_model", "rule_only", "model_deep"]


class DiagnosisEvidence(BaseModel):
    wrong_line: str
    correct_approach: str
    failed_test: str


class DiagnosisRecord(BaseModel):
    task_id: int
    fault_type: FaultType
    sub_tags: list[str]
    severity: Severity
    description: str
    evidence: DiagnosisEvidence
    enriched_labels: dict[str, str]
    confidence: float
    diagnosis_source: DiagnosisSource
    prompt_version: str
