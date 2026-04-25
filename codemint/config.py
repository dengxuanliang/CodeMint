from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator

from codemint.models.base import StrictModel


RetryBackoff = Literal["exponential"]
DifficultyDistribution = Literal["balanced", "weighted_hard"]
FaultType = Literal["comprehension", "modeling", "implementation", "edge_handling", "surface"]
Severity = Literal["low", "medium", "high"]
DifficultyLevel = Literal["medium", "hard"]


class ModelConfig(StrictModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    analysis_model: str | None = None
    evaluated_model: str | None = None
    temperature: float | None = None
    seed: int | None = None
    max_concurrency: int = 5
    max_retries: int = 3
    retry_backoff: RetryBackoff = "exponential"
    timeout: int = 120
    max_input_tokens: int = 8000


class EvaluationAPIConfig(StrictModel):
    base_url: str | None = None


class PromptConfig(StrictModel):
    override_dir: Path | None = None


class CustomPatternConfig(StrictModel):
    name: str
    pattern: str
    fault_type: FaultType
    sub_tag: str
    severity: Severity


class RulesConfig(StrictModel):
    custom_patterns: list[CustomPatternConfig] = Field(default_factory=list)
    disabled_rules: list[str] = Field(default_factory=list)
    severity_overrides: dict[str, Severity] = Field(default_factory=dict)
    rule_priority: list[str] = Field(default_factory=list)


class AggregateConfig(StrictModel):
    verification_level: Literal["auto", "exec", "cross-model", "self-check"] = "auto"
    max_cluster_size: int = 15
    sub_tag_limit_per_category: int = 20


class DiagnoseConfig(StrictModel):
    concurrency: int = 1


class NarrativeThemesConfig(StrictModel):
    generic: list[str] = Field(default_factory=list)
    domain_adaptive: bool = True


class SynthesizeConfig(StrictModel):
    specs_per_weakness: int = 3
    max_per_weakness: int = 8
    top_n: int = 10
    difficulty_levels: list[DifficultyLevel] = Field(default_factory=lambda: ["medium", "hard"])
    difficulty_distribution: DifficultyDistribution = "balanced"
    diversity_overlap_threshold: float = 0.5
    max_regeneration_attempts: int = 2
    narrative_themes: NarrativeThemesConfig = Field(default_factory=NarrativeThemesConfig)
    data_structures: list[str] = Field(
        default_factory=lambda: [
            "array",
            "tree",
            "graph",
            "string",
            "matrix",
            "heap",
            "stack",
            "hash_map",
        ]
    )

    @field_validator("difficulty_levels")
    @classmethod
    def reject_easy(cls, value: list[DifficultyLevel]) -> list[DifficultyLevel]:
        if "easy" in value:
            raise ValueError("easy is not allowed")
        return value


class CodeMintConfig(StrictModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation_api: EvaluationAPIConfig = Field(default_factory=EvaluationAPIConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    aggregate: AggregateConfig = Field(default_factory=AggregateConfig)
    diagnose: DiagnoseConfig = Field(default_factory=DiagnoseConfig)
    synthesize: SynthesizeConfig = Field(default_factory=SynthesizeConfig)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> CodeMintConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    expanded = _expand_env_vars(data)
    return CodeMintConfig.model_validate(expanded)
