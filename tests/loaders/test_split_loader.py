from __future__ import annotations

from pathlib import Path

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
