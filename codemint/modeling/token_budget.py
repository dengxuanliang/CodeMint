from __future__ import annotations

from dataclasses import replace

from codemint.models.task import TaskRecord


def truncate_payload(task: TaskRecord, max_input_tokens: int) -> TaskRecord:
    content_tokens = _estimate_tokens(task.content)
    if content_tokens >= max_input_tokens:
        return replace(task, test_code="", canonical_solution="")

    remaining_budget = max_input_tokens - content_tokens
    test_code = _truncate_text(task.test_code, remaining_budget)
    remaining_budget -= _estimate_tokens(test_code)
    canonical_solution = _truncate_text(task.canonical_solution, remaining_budget)
    return replace(task, test_code=test_code, canonical_solution=canonical_solution)


def _estimate_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return len(text.split())


def _truncate_text(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""

    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])

