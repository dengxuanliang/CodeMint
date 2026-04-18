from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from codemint.loaders import detect_loader
from codemint.models.task import TaskRecord


RunStage = Literal["diagnose", "aggregate", "synthesize"]
SECONDS_PER_MODEL_CALL = 3
AGGREGATE_TOKEN_COST = 4


@dataclass(frozen=True, slots=True)
class DryRunEstimate:
    input_count: int
    stage_calls: dict[str, int]
    stage_tokens: dict[str, int]
    stage_seconds: dict[str, int]
    estimated_model_calls: int
    estimated_tokens: int
    estimated_seconds: int

    @property
    def summary_line(self) -> str:
        return (
            f"Dry run total: {self.estimated_model_calls} model calls, "
            f"~{self.estimated_tokens} tokens, ~{self.estimated_seconds}s"
        )


def estimate_run(input_paths: list[Path], *, start_from: RunStage = "diagnose") -> DryRunEstimate:
    loader = detect_loader(input_paths)
    tasks = loader.load(input_paths)
    task_tokens = sum(_estimate_task_tokens(task) for task in tasks)
    stages = _selected_stages(start_from)
    stage_calls = {stage: 0 for stage in _all_stages()}
    stage_tokens = {stage: 0 for stage in _all_stages()}
    stage_seconds = {stage: 0 for stage in _all_stages()}

    if "diagnose" in stages:
        stage_calls["diagnose"] = len(tasks)
        stage_tokens["diagnose"] = task_tokens
    if "aggregate" in stages:
        stage_calls["aggregate"] = 1 if tasks else 0
        stage_tokens["aggregate"] = AGGREGATE_TOKEN_COST if tasks else 0
    if "synthesize" in stages:
        stage_calls["synthesize"] = len(tasks)
        stage_tokens["synthesize"] = task_tokens

    for stage, calls in stage_calls.items():
        stage_seconds[stage] = calls * SECONDS_PER_MODEL_CALL

    estimated_model_calls = sum(stage_calls.values())
    estimated_tokens = sum(stage_tokens.values())
    estimated_seconds = sum(stage_seconds.values())
    return DryRunEstimate(
        input_count=len(tasks),
        stage_calls=stage_calls,
        stage_tokens=stage_tokens,
        stage_seconds=stage_seconds,
        estimated_model_calls=estimated_model_calls,
        estimated_tokens=estimated_tokens,
        estimated_seconds=estimated_seconds,
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
