from __future__ import annotations

import json
from pathlib import Path

import pytest

from codemint.loaders.split import SplitFileLoader


def test_split_loader_joins_records_by_task_id() -> None:
    records = SplitFileLoader().load(
        [
            Path("tests/fixtures/input/split_inference.jsonl"),
            Path("tests/fixtures/input/split_results.jsonl"),
        ]
    )

    assert len(records) == 2
    assert records[0].task_id == 101
    assert records[0].accepted is False
    assert records[0].completion == "def solve():\n    return total_cost(items)"
    assert records[1].metrics["latency_ms"] == 3.1


def test_split_loader_raises_for_missing_result_row(tmp_path: Path) -> None:
    inference_path = tmp_path / "inference.jsonl"
    results_path = tmp_path / "results.jsonl"

    inference_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": 101,
                        "content": "Task 101",
                        "canonical_solution": "def solve():\n    return 1",
                        "completion": "def solve():\n    return 0",
                        "test_code": "assert solve() == 1",
                        "labels": {"difficulty": "hard"},
                        "extra": {"source": "fixture"},
                    }
                ),
                json.dumps(
                    {
                        "task_id": 102,
                        "content": "Task 102",
                        "canonical_solution": "def solve():\n    return 2",
                        "completion": "def solve():\n    return 2",
                        "test_code": "assert solve() == 2",
                        "labels": {"difficulty": "easy"},
                        "extra": {"source": "fixture"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        json.dumps({"task_id": 101, "accepted": False, "metrics": {"latency_ms": 1.0}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing result row for task_id 102"):
        SplitFileLoader().load([inference_path, results_path])


def test_split_loader_raises_for_duplicate_result_task_ids(tmp_path: Path) -> None:
    inference_path = tmp_path / "inference.jsonl"
    results_path = tmp_path / "results.jsonl"

    inference_path.write_text(
        json.dumps(
            {
                "task_id": 101,
                "content": "Task 101",
                "canonical_solution": "def solve():\n    return 1",
                "completion": "def solve():\n    return 0",
                "test_code": "assert solve() == 1",
                "labels": {"difficulty": "hard"},
                "extra": {"source": "fixture"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": 101, "accepted": False, "metrics": {"latency_ms": 1.0}}),
                json.dumps({"task_id": 101, "accepted": True, "metrics": {"latency_ms": 2.0}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate result row for task_id 101"):
        SplitFileLoader().load([inference_path, results_path])


def test_split_loader_raises_for_extra_result_row(tmp_path: Path) -> None:
    inference_path = tmp_path / "inference.jsonl"
    results_path = tmp_path / "results.jsonl"

    inference_path.write_text(
        json.dumps(
            {
                "task_id": 101,
                "content": "Task 101",
                "canonical_solution": "def solve():\n    return 1",
                "completion": "def solve():\n    return 0",
                "test_code": "assert solve() == 1",
                "labels": {"difficulty": "hard"},
                "extra": {"source": "fixture"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": 101, "accepted": False, "metrics": {"latency_ms": 1.0}}),
                json.dumps({"task_id": 999, "accepted": True, "metrics": {"latency_ms": 2.0}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra result row for task_id 999"):
        SplitFileLoader().load([inference_path, results_path])
