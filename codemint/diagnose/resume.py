from __future__ import annotations

from pathlib import Path

from codemint.io.jsonl import read_jsonl


def find_missing_task_ids(existing_diagnoses_path: Path, expected_task_ids: list[int]) -> list[int]:
    if not existing_diagnoses_path.exists():
        return list(expected_task_ids)

    completed_task_ids: set[int] = set()
    for line_number, row in enumerate(read_jsonl(existing_diagnoses_path), start=1):
        if "task_id" not in row:
            raise ValueError(
                f"Missing task_id in {existing_diagnoses_path} at line {line_number}"
            )

        task_id = row["task_id"]
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError(
                f"Invalid task_id in {existing_diagnoses_path} at line {line_number}: "
                f"{task_id!r}"
            )

        completed_task_ids.add(task_id)

    return [task_id for task_id in expected_task_ids if task_id not in completed_task_ids]
