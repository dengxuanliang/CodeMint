from __future__ import annotations

import json
from pathlib import Path

import pytest

from codemint.loaders.detect import detect_loader
from codemint.loaders.merged import MergedFileLoader
from codemint.loaders.real_log import RealLogFileLoader
from codemint.loaders.split import SplitFileLoader


def test_detects_merged_loader_for_single_file() -> None:
    loader = detect_loader([Path("tests/fixtures/input/merged_eval.jsonl")])

    assert isinstance(loader, MergedFileLoader)


def test_detects_real_log_loader_for_single_file() -> None:
    loader = detect_loader([Path("tests/fixtures/input/real_log_eval.jsonl")])

    assert isinstance(loader, RealLogFileLoader)


def test_detects_split_loader_for_inference_and_results_files() -> None:
    loader = detect_loader(
        [
            Path("tests/fixtures/input/split_inference.jsonl"),
            Path("tests/fixtures/input/split_results.jsonl"),
        ]
    )

    assert isinstance(loader, SplitFileLoader)


def test_rejects_split_detection_for_underspecified_inference_file(tmp_path: Path) -> None:
    inference_path = tmp_path / "inference.jsonl"
    results_path = tmp_path / "results.jsonl"

    inference_path.write_text(
        json.dumps({"task_id": 101, "completion": "def solve():\n    return 1"}) + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        json.dumps({"task_id": 101, "accepted": False, "metrics": {"latency_ms": 2.4}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not detect input loader"):
        detect_loader([inference_path, results_path])


def test_rejects_split_detection_for_two_merged_files() -> None:
    with pytest.raises(ValueError, match="Could not detect input loader"):
        detect_loader(
            [
                Path("tests/fixtures/input/merged_eval.jsonl"),
                Path("tests/fixtures/input/merged_eval.jsonl"),
            ]
        )
