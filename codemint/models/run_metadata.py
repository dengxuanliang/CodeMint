from __future__ import annotations

from datetime import datetime
from typing import Any

from codemint.models.base import StrictModel


class PromptVersions(StrictModel):
    diagnose: str
    aggregate: str
    synthesize: str


class RunSummary(StrictModel):
    diagnosed: int
    rule_screened: int
    model_analyzed: int
    errors: int
    weaknesses_found: int
    specs_generated: int


class RunMetadata(StrictModel):
    run_id: str
    timestamp: datetime
    config_snapshot: dict[str, Any]
    analysis_model: str
    prompt_versions: PromptVersions
    input_files: list[str]
    input_count: int
    stages_executed: list[str]
    summary: RunSummary
