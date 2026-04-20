from __future__ import annotations

from pathlib import Path

import pytest

from codemint.io.jsonl import append_jsonl, read_jsonl


def test_append_jsonl_writes_rows_as_utf8_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "diagnoses.jsonl"

    append_jsonl(path, [{"task_id": 1, "description": "naive caf\u00e9"}])

    assert path.read_text(encoding="utf-8") == '{"task_id": 1, "description": "naive café"}\n'


def test_read_jsonl_returns_rows_in_order(tmp_path: Path) -> None:
    path = tmp_path / "diagnoses.jsonl"
    path.write_text('{"task_id": 1}\n{"task_id": 2}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"task_id": 1}, {"task_id": 2}]


def test_append_jsonl_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "outputs" / "run-123" / "diagnoses.jsonl"

    append_jsonl(path, [{"task_id": 1}])

    assert path.exists()


def test_read_jsonl_raises_clear_error_for_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "diagnoses.jsonl"
    path.write_text('{"task_id": 1}\n{"task_id": }\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"diagnoses\.jsonl.*line 2"):
        read_jsonl(path)


def test_read_jsonl_rejects_non_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "diagnoses.jsonl"
    path.write_text('{"task_id": 1}\n[1, 2, 3]\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"diagnoses\.jsonl.*line 2"):
        read_jsonl(path)


def test_append_jsonl_appends_instead_of_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "diagnoses.jsonl"

    append_jsonl(path, [{"task_id": 1}])
    append_jsonl(path, [{"task_id": 2}])

    assert read_jsonl(path) == [{"task_id": 1}, {"task_id": 2}]
