from __future__ import annotations

from codemint.modeling.token_budget import truncate_payload
from codemint.models.task import TaskRecord


def make_task() -> TaskRecord:
    return TaskRecord(
        task_id=1,
        content="problem statement " * 10,
        canonical_solution="solution " * 20,
        completion="completion",
        test_code="test " * 30,
        labels={"topic": "arrays"},
        accepted=True,
        metrics={"score": 1},
        extra={"source": "fixture"},
    )


def test_truncate_payload_preserves_content_and_trims_tests_first() -> None:
    task = make_task()

    truncated = truncate_payload(task, max_input_tokens=40)

    assert truncated is not task
    assert truncated.content == task.content
    assert len(truncated.test_code) < len(task.test_code)
    assert len(truncated.canonical_solution) <= len(task.canonical_solution)


def test_truncate_payload_does_not_mutate_original_task() -> None:
    task = make_task()
    original = (task.content, task.test_code, task.canonical_solution)

    truncate_payload(task, max_input_tokens=40)

    assert (task.content, task.test_code, task.canonical_solution) == original

