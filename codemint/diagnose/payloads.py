from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from codemint.config import CodeMintConfig
from codemint.models.task import TaskRecord


@dataclass(frozen=True, slots=True)
class DiagnosePayloadView:
    content: str
    completion: str
    canonical_solution: str
    test_code: str
    completion_truncated: bool
    canonical_solution_truncated: bool
    test_code_truncated: bool


def build_diagnose_payload(task: TaskRecord, *, config: CodeMintConfig) -> dict:
    budgeted = budget_diagnose_task(task, config.model.max_input_tokens)
    return {
        "task_id": task.task_id,
        "content": budgeted.content,
        "completion": budgeted.completion,
        "canonical_solution": budgeted.canonical_solution,
        "test_code": budgeted.test_code,
        "labels": task.labels,
        "accepted": task.accepted,
        "programming_language": str(task.labels.get("programming_language", "")),
        "execution_language": str(task.labels.get("execution_language", "")),
        "prompt_requests_markdown_wrapper": prompt_requests_markdown_wrapper(task.content),
        "truncation_info": {
            "completion_truncated": budgeted.completion_truncated,
            "canonical_solution_truncated": budgeted.canonical_solution_truncated,
            "test_code_truncated": budgeted.test_code_truncated,
        },
    }


def budget_diagnose_task(task: TaskRecord, max_input_tokens: int) -> DiagnosePayloadView:
    content = task.content
    completion = task.completion
    canonical_solution = task.canonical_solution
    test_code = task.test_code

    completion_truncated = False
    canonical_solution_truncated = False
    test_code_truncated = False

    if _payload_tokens(content, completion, canonical_solution, test_code) <= max_input_tokens:
        return DiagnosePayloadView(
            content=content,
            completion=completion,
            canonical_solution=canonical_solution,
            test_code=test_code,
            completion_truncated=completion_truncated,
            canonical_solution_truncated=canonical_solution_truncated,
            test_code_truncated=test_code_truncated,
        )

    if canonical_solution:
        canonical_solution = ""
        canonical_solution_truncated = True

    if _payload_tokens(content, completion, canonical_solution, test_code) > max_input_tokens and test_code:
        test_code = _truncate_test_code(test_code, max(40, max_input_tokens // 4))
        test_code_truncated = True

    if _payload_tokens(content, completion, canonical_solution, test_code) > max_input_tokens and completion:
        completion = _truncate_head_tail(completion, max(80, max_input_tokens // 3))
        completion_truncated = True

    if _payload_tokens(content, completion, canonical_solution, test_code) > max_input_tokens and content:
        content = _truncate_head_tail(content, max(80, max_input_tokens // 3))

    return DiagnosePayloadView(
        content=content,
        completion=completion,
        canonical_solution=canonical_solution,
        test_code=test_code,
        completion_truncated=completion_truncated,
        canonical_solution_truncated=canonical_solution_truncated,
        test_code_truncated=test_code_truncated,
    )


def prompt_requests_markdown_wrapper(content: str) -> bool:
    lowered = content.lower()
    if "markdown" not in lowered:
        return False
    if "```" in content:
        return True
    return bool(
        re.search(
            r"(wrap|enclose).*(markdown).*code block|markdown.*wrapper|code block format",
            lowered,
        )
    )


def _payload_tokens(content: str, completion: str, canonical_solution: str, test_code: str) -> int:
    return sum(_estimate_tokens(part) for part in (content, completion, canonical_solution, test_code))


def _estimate_tokens(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _truncate_test_code(text: str, max_tokens: int) -> str:
    insert_index = text.find("#<INSERT>")
    if insert_index == -1:
        return _truncate_head_tail(text, max_tokens)

    prefix = text[:insert_index]
    suffix = text[insert_index:]
    prefix_tokens = prefix.split()
    suffix_tokens = suffix.split()
    kept_prefix = prefix_tokens[-max_tokens // 3 :] if prefix_tokens else []
    kept_suffix = suffix_tokens[: max_tokens - len(kept_prefix)] if suffix_tokens else []
    return " ".join([*kept_prefix, *kept_suffix])


def _truncate_head_tail(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    head_count = max_tokens // 2
    tail_count = max_tokens - head_count
    return " ".join([*tokens[:head_count], "...", *tokens[-tail_count:]])
