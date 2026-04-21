from __future__ import annotations

from dataclasses import dataclass
import re

from codemint.models.task import TaskRecord


_WHITESPACE_RE = re.compile(r"\s+")
_FUNCTION_DEF_RE = re.compile(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_ASSERT_CALL_RE = re.compile(r"assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*==\s*(.+)")


@dataclass(frozen=True, slots=True)
class DiagnoseFingerprint:
    task_id: int
    rule_hint: str | None
    entry_point_hint: str | None
    output_format_hint: str | None
    assertion_hint: str | None
    syntax_hint: str | None
    normalized_completion: str


def build_fingerprint(task: TaskRecord, *, rule_hint: str | None = None) -> DiagnoseFingerprint:
    normalized_completion = _normalize_completion(task.completion)
    entry_point_hint = _extract_entry_point_hint(task.completion)
    output_format_hint = _extract_output_format_hint(task.completion)
    assertion_hint = _extract_assertion_hint(task.test_code)
    syntax_hint = _extract_syntax_hint(task.completion)

    return DiagnoseFingerprint(
        task_id=task.task_id,
        rule_hint=rule_hint,
        entry_point_hint=entry_point_hint,
        output_format_hint=output_format_hint,
        assertion_hint=assertion_hint,
        syntax_hint=syntax_hint,
        normalized_completion=normalized_completion,
    )


def _normalize_completion(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    collapsed = "\n".join(lines)
    return _WHITESPACE_RE.sub(" ", collapsed).strip()


def _extract_entry_point_hint(text: str) -> str | None:
    match = _FUNCTION_DEF_RE.search(text)
    if match is None:
        return None
    return match.group(1)


def _extract_output_format_hint(text: str) -> str | None:
    lowered = text.lower()
    if "```" in lowered:
        return "markdown_fence"
    return "raw_code"


def _extract_assertion_hint(test_code: str) -> str | None:
    match = _ASSERT_CALL_RE.search(test_code)
    if match is None:
        return None
    function_name = match.group(1)
    return f"assert {function_name}(?) == ?"


def _extract_syntax_hint(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*(?:\n|$)", lowered) and ":" not in lowered.splitlines()[0]:
        return "missing_colon"
    return None
