from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from codemint.config import CodeMintConfig
from codemint.loaders import detect_loader
from codemint.models.task import TaskRecord


RunStage = Literal["diagnose", "aggregate", "synthesize"]
SECONDS_PER_MODEL_CALL = 3
AGGREGATE_TOKEN_COST = 4


@dataclass(frozen=True, slots=True)
class DryRunEstimate:
    input_count: int
    concurrency: int
    stage_calls: dict[str, int]
    stage_input_tokens: dict[str, int]
    stage_output_tokens: dict[str, int]
    stage_seconds: dict[str, int]
    estimated_model_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_seconds: int
    rule_screened: int

    @property
    def summary_line(self) -> str:
        return (
            f"Estimated model calls: {self.estimated_model_calls} "
            f"(diagnose: {self.stage_calls['diagnose']}, aggregate: {self.stage_calls['aggregate']}, "
            f"synthesize: {self.stage_calls['synthesize']})"
        )


def estimate_run(
    input_paths: list[Path],
    *,
    start_from: RunStage = "diagnose",
    config: CodeMintConfig | None = None,
) -> DryRunEstimate:
    resolved_config = config or CodeMintConfig()
    loader = detect_loader(input_paths)
    tasks = loader.load(input_paths)
    task_tokens = sum(_estimate_task_tokens(task) for task in tasks)
    stages = _selected_stages(start_from)
    stage_calls = {stage: 0 for stage in _all_stages()}
    stage_input_tokens = {stage: 0 for stage in _all_stages()}
    stage_output_tokens = {stage: 0 for stage in _all_stages()}
    stage_seconds = {stage: 0 for stage in _all_stages()}
    rule_screened = _estimate_rule_screened(tasks) if "diagnose" in stages else 0

    if "diagnose" in stages:
        stage_calls["diagnose"] = max(len(tasks) - rule_screened, 0)
        stage_input_tokens["diagnose"] = task_tokens
        stage_output_tokens["diagnose"] = stage_calls["diagnose"] * 32
    if "aggregate" in stages:
        stage_calls["aggregate"] = 1 if tasks else 0
        stage_input_tokens["aggregate"] = AGGREGATE_TOKEN_COST if tasks else 0
        stage_output_tokens["aggregate"] = 16 if tasks else 0
    if "synthesize" in stages:
        stage_calls["synthesize"] = len(tasks)
        stage_input_tokens["synthesize"] = task_tokens
        stage_output_tokens["synthesize"] = len(tasks) * 24

    for stage, calls in stage_calls.items():
        stage_seconds[stage] = calls * SECONDS_PER_MODEL_CALL

    estimated_model_calls = sum(stage_calls.values())
    estimated_input_tokens = sum(stage_input_tokens.values())
    estimated_output_tokens = sum(stage_output_tokens.values())
    concurrency = max(resolved_config.model.max_concurrency, 1)
    estimated_seconds = max(1, sum(stage_seconds.values()) // concurrency) if estimated_model_calls else 0
    return DryRunEstimate(
        input_count=len(tasks),
        concurrency=concurrency,
        stage_calls=stage_calls,
        stage_input_tokens=stage_input_tokens,
        stage_output_tokens=stage_output_tokens,
        stage_seconds=stage_seconds,
        estimated_model_calls=estimated_model_calls,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_seconds=estimated_seconds,
        rule_screened=rule_screened,
    )


def _estimate_task_tokens(task: TaskRecord) -> int:
    return (
        _count_words(task.content)
        + _count_words(task.completion)
        + _count_words(task.test_code)
        + _count_words(task.canonical_solution)
    )


def _count_words(value: str) -> int:
    return len(value.split())


def _selected_stages(start_from: RunStage) -> tuple[RunStage, ...]:
    stage_order: tuple[RunStage, ...] = _all_stages()
    start_index = stage_order.index(start_from)
    return stage_order[start_index:]


def _all_stages() -> tuple[RunStage, ...]:
    return ("diagnose", "aggregate", "synthesize")


def _estimate_rule_screened(tasks: list[TaskRecord]) -> int:
    screened = 0
    for task in tasks:
        text = task.completion.lower()
        if any(token in text for token in ("syntaxerror", "nameerror", "indexerror", "keyerror", "importerror")):
            screened += 1
    return screened
