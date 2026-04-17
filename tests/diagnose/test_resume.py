from __future__ import annotations

from pathlib import Path

from codemint.diagnose.resume import find_missing_task_ids


def test_missing_task_ids_are_detected_from_existing_diagnoses(tmp_path: Path) -> None:
    existing = tmp_path / "diagnoses.jsonl"
    existing.write_text('{"task_id": 1}\n{"task_id": 3}\n', encoding="utf-8")

    assert find_missing_task_ids(existing, [1, 2, 3]) == [2]


def test_missing_task_ids_returns_all_expected_ids_when_file_is_absent(tmp_path: Path) -> None:
    existing = tmp_path / "diagnoses.jsonl"

    assert find_missing_task_ids(existing, [1, 2, 3]) == [1, 2, 3]
