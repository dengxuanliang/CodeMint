from __future__ import annotations

from codemint.models.task import TaskRecord


def test_fingerprint_captures_entry_point_and_format_hints() -> None:
    from codemint.diagnose.fingerprint import build_fingerprint

    task = _task(
        1,
        completion="```python\ndef solve(x)\n    return x + 1\n```",
        test_code="assert solve(1) == 2",
    )

    fingerprint = build_fingerprint(task)

    assert fingerprint.entry_point_hint == "solve"
    assert fingerprint.output_format_hint == "markdown_fence"
    assert fingerprint.assertion_hint == "assert solve(?) == ?"
    assert fingerprint.syntax_hint == "missing_colon"


def test_fingerprint_prefers_explicit_rule_hint_when_provided() -> None:
    from codemint.diagnose.fingerprint import build_fingerprint

    task = _task(
        2,
        completion="def solve_value(x):\n    return x + 1",
        test_code="assert solve(1) == 2",
    )

    fingerprint = build_fingerprint(task, rule_hint="function_name_mismatch")

    assert fingerprint.rule_hint == "function_name_mismatch"
    assert fingerprint.entry_point_hint == "solve_value"


def test_fingerprint_normalizes_completion_whitespace() -> None:
    from codemint.diagnose.fingerprint import build_fingerprint

    compact = _task(
        3,
        completion="def solve(x):\n    return x + 1\n",
        test_code="assert solve(1) == 2",
    )
    noisy = _task(
        4,
        completion="\n\n def solve(x):\n\treturn x + 1   \n",
        test_code="assert solve(1) == 2",
    )

    compact_fingerprint = build_fingerprint(compact)
    noisy_fingerprint = build_fingerprint(noisy)

    assert compact_fingerprint.normalized_completion == noisy_fingerprint.normalized_completion


def _task(task_id: int, *, completion: str, test_code: str) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        content="Implement solve(x).",
        canonical_solution="def solve(x):\n    return x + 1",
        completion=completion,
        test_code=test_code,
        labels={},
        accepted=False,
        metrics={},
        extra={},
    )
