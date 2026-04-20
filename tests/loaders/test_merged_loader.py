from __future__ import annotations

from pathlib import Path

from codemint.loaders.merged import MergedFileLoader


def test_merged_loader_parses_task_records() -> None:
    records = MergedFileLoader().load([Path("tests/fixtures/input/merged_eval.jsonl")])

    assert len(records) == 2
    assert records[0].task_id == 101
    assert records[0].accepted is False
    assert records[1].labels["difficulty"] == "easy"
