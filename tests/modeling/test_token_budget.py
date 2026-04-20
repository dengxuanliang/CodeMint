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
    assert truncated.canonical_solution == task.canonical_solution


def test_truncate_payload_does_not_mutate_original_task() -> None:
    task = make_task()
    original = (task.content, task.test_code, task.canonical_solution)

    truncate_payload(task, max_input_tokens=40)

    assert (task.content, task.test_code, task.canonical_solution) == original


def test_truncate_payload_preserves_canonical_solution_before_trimming_it() -> None:
    task = make_task()
    content_tokens = len(task.content.split())
    canonical_tokens = len(task.canonical_solution.split())

    truncated = truncate_payload(
        task,
        max_input_tokens=content_tokens + canonical_tokens + 5,
    )

    assert truncated.content == task.content
    assert truncated.canonical_solution == task.canonical_solution
    assert truncated.test_code.split() == ["test"] * 5


def test_truncate_payload_trims_canonical_solution_only_after_tests_are_empty() -> None:
    task = make_task()
    content_tokens = len(task.content.split())

    truncated = truncate_payload(task, max_input_tokens=content_tokens + 10)

    assert truncated.content == task.content
    assert truncated.test_code == ""
    assert truncated.canonical_solution.split() == ["solution"] * 10
