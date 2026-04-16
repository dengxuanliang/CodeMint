from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class PromptVersions(BaseModel):
    diagnose: str
    aggregate: str
    synthesize: str


class RunSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    diagnosed: int
    rule_screened: int
    model_analyzed: int
    errors: int
    weaknesses_found: int
    specs_generated: int


class RunMetadata(BaseModel):
    run_id: str
    timestamp: datetime
    config_snapshot: dict[str, Any]
    analysis_model: str
    prompt_versions: PromptVersions
    input_files: list[str]
    input_count: int
    stages_executed: list[str]
    summary: RunSummary
