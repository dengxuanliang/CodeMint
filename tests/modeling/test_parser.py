from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from codemint.modeling.parser import parse_with_retry
from codemint.models.base import StrictModel


class AnswerModel(StrictModel):
    answer: str


def test_parser_retries_once_on_schema_failure() -> None:
    responses = iter(['{"wrong":"field"}', '{"answer":"ok"}'])
    call_contexts: list[str | None] = []

    def invoke(format_error: str | None = None) -> str:
        call_contexts.append(format_error)
        return next(responses)

    parsed = parse_with_retry(AnswerModel, invoke)

    assert parsed == AnswerModel(answer="ok")
    assert call_contexts[0] is None
    assert call_contexts[1] is not None
    assert "answer" in call_contexts[1]


def test_parser_raises_after_retry_exhausted() -> None:
    def invoke(format_error: str | None = None) -> str:
        return '{"wrong":"field"}'

    with pytest.raises(ValidationError):
        parse_with_retry(AnswerModel, invoke)


def test_parser_retries_once_on_invalid_json() -> None:
    responses = iter(['{"answer":', '{"answer":"ok"}'])
    call_contexts: list[str | None] = []

    def invoke(format_error: str | None = None) -> str:
        call_contexts.append(format_error)
        return next(responses)

    parsed = parse_with_retry(AnswerModel, invoke)

    assert parsed == AnswerModel(answer="ok")
    assert call_contexts[0] is None
    assert call_contexts[1] is not None
