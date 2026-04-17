from __future__ import annotations

from pathlib import Path

from codemint.loaders.detect import detect_loader


def test_detects_merged_loader_for_single_file() -> None:
    loader = detect_loader([Path("tests/fixtures/input/merged_eval.jsonl")])

    assert loader.__class__.__name__ == "MergedFileLoader"


def test_detects_split_loader_for_inference_and_results_files() -> None:
    loader = detect_loader(
        [
            Path("tests/fixtures/input/split_inference.jsonl"),
            Path("tests/fixtures/input/split_results.jsonl"),
        ]
    )

    assert loader.__class__.__name__ == "SplitFileLoader"
