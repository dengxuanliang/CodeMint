from __future__ import annotations

from codemint.config import CodeMintConfig
from codemint.diagnose.payloads import build_diagnose_payload
from codemint.models.task import TaskRecord


def _task(**overrides) -> TaskRecord:
    payload = dict(
        task_id=101,
        content="Please wrap the answering code with markdown code block format, e.g. ```html [code] ```",
        canonical_solution="def solve():\n    return 1",
        completion="```html\n<form></form>\n```",
        test_code="#<INSERT>\nassert True\n" + ("tail " * 200),
        labels={
            "programming_language": "html",
            "execution_language": "jest",
            "difficulty": "medium",
        },
        accepted=False,
        metrics={},
        extra={},
    )
    payload.update(overrides)
    return TaskRecord(**payload)


def test_build_diagnose_payload_surfaces_language_and_markdown_contract() -> None:
    payload = build_diagnose_payload(
        _task(),
        config=CodeMintConfig.model_validate({"model": {"max_input_tokens": 8000}}),
    )

    assert payload["programming_language"] == "html"
    assert payload["execution_language"] == "jest"
    assert payload["prompt_requests_markdown_wrapper"] is True


def test_build_diagnose_payload_omits_metrics_and_surfaces_truncation_info() -> None:
    task = _task(
        completion=("head " * 300) + "#<INSERT>\n" + ("tail " * 300),
        canonical_solution="solution " * 200,
        test_code=("setup " * 300) + "#<INSERT>\nassert True\n" + ("check " * 300),
    )

    payload = build_diagnose_payload(
        task,
        config=CodeMintConfig.model_validate({"model": {"max_input_tokens": 80}}),
    )

    assert "metrics" not in payload
    assert payload["truncation_info"]["canonical_solution_truncated"] is True
    assert payload["truncation_info"]["test_code_truncated"] is True
    assert payload["truncation_info"]["completion_truncated"] is True
    assert "#<INSERT>" in payload["test_code"]


def test_build_diagnose_payload_does_not_require_markdown_when_prompt_does_not_ask_for_it() -> None:
    payload = build_diagnose_payload(
        _task(content="Return raw executable Python code only."),
        config=CodeMintConfig.model_validate({"model": {"max_input_tokens": 8000}}),
    )

    assert payload["prompt_requests_markdown_wrapper"] is False
