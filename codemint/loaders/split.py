from __future__ import annotations

from pathlib import Path
from typing import Any

from codemint.loaders.base import BaseLoader, read_jsonl
from codemint.models.task import TaskRecord


class SplitFileLoader(BaseLoader):
    def load(self, paths: list[Path]) -> list[TaskRecord]:
        if len(paths) != 2:
            raise ValueError("SplitFileLoader expects inference and results files")

        first_records = read_jsonl(paths[0])
        second_records = read_jsonl(paths[1])
        inference_records, result_records = self._partition_records(first_records, second_records)
        results_by_task_id = {record["task_id"]: record for record in result_records}

        return [
            TaskRecord(**{**inference, **results_by_task_id[inference["task_id"]]})
            for inference in inference_records
        ]

    def _partition_records(
        self,
        first_records: list[dict[str, Any]],
        second_records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if first_records and "accepted" in first_records[0]:
            return second_records, first_records
        return first_records, second_records
