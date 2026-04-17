from __future__ import annotations

from pathlib import Path

import pytest

from codemint.diagnose.resume import find_missing_task_ids


def test_missing_task_ids_are_detected_from_existing_diagnoses(tmp_path: Path) -> None:
    existing = tmp_path / "diagnoses.jsonl"
    existing.write_text('{"task_id": 1}\n{"task_id": 3}\n', encoding="utf-8")

    assert find_missing_task_ids(existing, [1, 2, 3]) == [2]


def test_missing_task_ids_returns_all_expected_ids_when_file_is_absent(tmp_path: Path) -> None:
    existing = tmp_path / "diagnoses.jsonl"

    assert find_missing_task_ids(existing, [1, 2, 3]) == [1, 2, 3]


def test_find_missing_task_ids_raises_for_missing_task_id(tmp_path: Path) -> None:
    existing = tmp_path / "diagnoses.jsonl"
    existing.write_text('{"task_id": 1}\n{"description": "missing"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"task_id.*line 2"):
        find_missing_task_ids(existing, [1, 2, 3])


def test_find_missing_task_ids_raises_for_invalid_task_id(tmp_path: Path) -> None:
    existing = tmp_path / "diagnoses.jsonl"
    existing.write_text('{"task_id": 1}\n{"task_id": "abc"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"task_id.*line 2"):
        find_missing_task_ids(existing, [1, 2, 3])


@pytest.mark.parametrize(
    ("raw_task_id", "expected_pattern"),
    [
        ("true", r"task_id.*line 2"),
        ("1.9", r"task_id.*line 2"),
        ('"2"', r"task_id.*line 2"),
    ],
)
def test_find_missing_task_ids_rejects_non_integer_task_id_types(
    tmp_path: Path, raw_task_id: str, expected_pattern: str
) -> None:
    existing = tmp_path / "diagnoses.jsonl"
    existing.write_text(
        f'{{"task_id": 1}}\n{{"task_id": {raw_task_id}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_pattern):
        find_missing_task_ids(existing, [1, 2, 3])
