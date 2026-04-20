from __future__ import annotations

import re
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def parse_with_retry(
    schema: type[SchemaT],
    invoke: Callable[[str | None], str],
) -> SchemaT:
    try:
        return schema.model_validate_json(_normalize_json_text(invoke(None)))
    except ValidationError as first_error:
        format_error = _build_format_error(schema, first_error)
        return schema.model_validate_json(_normalize_json_text(invoke(format_error)))


def _build_format_error(schema: type[BaseModel], error: ValidationError) -> str:
    return (
        f"Response did not match schema {schema.__name__}. "
        f"Return valid JSON matching the schema. Error: {error}"
    )


def _normalize_json_text(raw: str) -> str:
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text
