from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codemint.loaders.base import BaseLoader, read_jsonl
from codemint.models.task import TaskRecord


class RealLogFileLoader(BaseLoader):
    def load(self, paths: list[Path]) -> list[TaskRecord]:
        if len(paths) != 1:
            raise ValueError("RealLogFileLoader expects exactly one input file")

        return [self._to_task_record(record) for record in read_jsonl(paths[0])]

    def _to_task_record(self, record: dict[str, Any]) -> TaskRecord:
        test = record.get("test") or {}
        test_code = _test_code(test)
        pass_at_1 = record.get("pass_at_1")
        extra = {
            "model": record.get("model"),
        }

        return TaskRecord(
            task_id=int(record["id"]),
            content=str(record.get("content", "")),
            canonical_solution=str(record.get("canonical_solution", "")),
            completion=str(record.get("completion", "")),
            test_code=test_code,
            labels=_filtered_labels(record.get("labels")),
            accepted=pass_at_1 == 1,
            metrics={},
            extra=extra,
        )


def _test_code(test: Any) -> str:
    if isinstance(test, dict):
        code = test.get("code", "")
        if isinstance(code, str):
            return code
        return json.dumps(code, ensure_ascii=False)
    if isinstance(test, str):
        return test
    return json.dumps(test, ensure_ascii=False)


def _filtered_labels(labels: Any) -> dict[str, Any]:
    if not isinstance(labels, dict):
        return {}
    return {
        key: value
        for key, value in labels.items()
        if key not in {"fewshot", "locale", "is_lctx"}
    }
