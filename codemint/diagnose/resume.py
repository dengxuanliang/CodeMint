from __future__ import annotations

from pathlib import Path

from codemint.io.jsonl import read_jsonl


def find_missing_task_ids(existing_diagnoses_path: Path, expected_task_ids: list[int]) -> list[int]:
    if not existing_diagnoses_path.exists():
        return list(expected_task_ids)

    completed_task_ids = {
        int(row["task_id"])
        for row in read_jsonl(existing_diagnoses_path)
        if "task_id" in row
    }
    return [task_id for task_id in expected_task_ids if task_id not in completed_task_ids]
