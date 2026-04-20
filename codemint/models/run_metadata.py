from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from codemint.models.base import StrictModel


class PromptVersions(StrictModel):
    diagnose: str
    aggregate: str
    synthesize: str


SynthesizeStatus = Literal["skipped", "success", "degraded"]


class RunSummary(StrictModel):
    diagnosed: int
    rule_screened: int
    model_analyzed: int
    non_failures: int
    errors: int
    skipped: int
    elapsed_seconds: float
    weaknesses_found: int
    specs_generated: int
    synthesize_failures: int
    specs_by_weakness: dict[str, int]
    synthesize_status: SynthesizeStatus
    attempted_weaknesses: list[str] = []
    covered_weaknesses: list[str] = []
    weaknesses_without_specs: list[str]
    synthesize_failure_reasons_by_weakness: dict[str, list[str]] = {}


class RunMetadata(StrictModel):
    run_id: str
    timestamp: datetime
    config_snapshot: dict[str, Any]
    analysis_model: str
    prompt_versions: PromptVersions
    input_files: list[str]
    input_count: int
    stages_executed: list[str]
    self_analysis_warning: bool = False
    summary: RunSummary


class RunProgressEvent(StrictModel):
    stage: str
    status: str
    processed: int
    total: int
    errors: int
    eta_seconds: int | None = None
