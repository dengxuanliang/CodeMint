from __future__ import annotations

from pathlib import Path

from codemint.loaders.base import BaseLoader, read_jsonl
from codemint.loaders.merged import MergedFileLoader
from codemint.loaders.split import SplitFileLoader

TASK_FIELDS = {
    "task_id",
    "content",
    "canonical_solution",
    "completion",
    "test_code",
    "labels",
    "accepted",
    "metrics",
    "extra",
}


def detect_loader(paths: list[Path]) -> BaseLoader:
    if len(paths) == 1:
        first_record = _first_record(paths[0])
        if TASK_FIELDS.issubset(first_record):
            return MergedFileLoader()

    if len(paths) == 2:
        fields = [set(_first_record(path)) for path in paths]
        has_inference = any({"task_id", "completion"}.issubset(field) for field in fields)
        has_results = any({"task_id", "accepted", "metrics"}.issubset(field) for field in fields)
        if has_inference and has_results:
            return SplitFileLoader()

    raise ValueError("Could not detect input loader for provided paths")


def _first_record(path: Path) -> dict[str, object]:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"Input file is empty: {path}")
    return records[0]
