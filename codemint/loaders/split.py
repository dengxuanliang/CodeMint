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
        results_by_task_id = self._index_results(result_records)
        inference_task_ids = [record["task_id"] for record in inference_records]
        inference_task_id_set = set(inference_task_ids)

        for task_id in inference_task_ids:
            if task_id not in results_by_task_id:
                raise ValueError(f"Missing result row for task_id {task_id}")

        extra_result_ids = set(results_by_task_id) - inference_task_id_set
        if extra_result_ids:
            extra_task_id = min(extra_result_ids)
            raise ValueError(f"Extra result row for task_id {extra_task_id}")

        return [TaskRecord(**{**inference, **results_by_task_id[inference["task_id"]]}) for inference in inference_records]

    def _partition_records(
        self,
        first_records: list[dict[str, Any]],
        second_records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if first_records and "accepted" in first_records[0]:
            return second_records, first_records
        return first_records, second_records

    def _index_results(self, result_records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        results_by_task_id: dict[int, dict[str, Any]] = {}
        for record in result_records:
            task_id = record["task_id"]
            if task_id in results_by_task_id:
                raise ValueError(f"Duplicate result row for task_id {task_id}")
            results_by_task_id[task_id] = record
        return results_by_task_id
