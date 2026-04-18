from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codemint.loaders import detect_loader
from codemint.models.task import TaskRecord


SECONDS_PER_MODEL_CALL = 3


@dataclass(frozen=True, slots=True)
class DryRunEstimate:
    input_count: int
    stage_calls: dict[str, int]
    estimated_model_calls: int
    estimated_tokens: int
    estimated_seconds: int

    @property
    def summary_line(self) -> str:
        return (
            f"Dry run: {self.estimated_model_calls} model calls, "
            f"~{self.estimated_tokens} tokens, ~{self.estimated_seconds}s"
        )


def estimate_run(input_paths: list[Path]) -> DryRunEstimate:
    loader = detect_loader(input_paths)
    tasks = loader.load(input_paths)
    stage_calls = {
        "diagnose": len(tasks),
        "aggregate": 1 if tasks else 0,
        "synthesize": len(tasks),
    }
    estimated_model_calls = sum(stage_calls.values())
    input_tokens = sum(_estimate_task_tokens(task) for task in tasks)
    estimated_tokens = input_tokens * 2 + stage_calls["aggregate"] * 4
    estimated_seconds = estimated_model_calls * SECONDS_PER_MODEL_CALL
    return DryRunEstimate(
        input_count=len(tasks),
        stage_calls=stage_calls,
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
