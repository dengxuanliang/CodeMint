from __future__ import annotations

from pathlib import Path

from codemint.run.dry_run import estimate_run
from codemint.logging import format_dry_run_summary


def test_dry_run_reports_estimated_calls_tokens_and_time(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                '{"task_id": 1, "content": "a b c d", "canonical_solution": "ok", "completion": "x y", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
                '{"task_id": 2, "content": "alpha beta", "canonical_solution": "ok", "completion": "theta", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
                '{"task_id": 3, "content": "one two three", "canonical_solution": "ok", "completion": "four five six", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    estimate = estimate_run([input_path])

    assert estimate.input_count == 3
    assert estimate.stage_calls == {"diagnose": 3, "aggregate": 1, "synthesize": 3}
    assert estimate.stage_tokens == {"diagnose": 24, "aggregate": 4, "synthesize": 24}
    assert estimate.stage_seconds == {"diagnose": 9, "aggregate": 3, "synthesize": 9}
    assert estimate.estimated_model_calls == 7
    assert estimate.estimated_tokens == 52
    assert estimate.estimated_seconds == 21
    assert estimate.summary_line == "Dry run total: 7 model calls, ~52 tokens, ~21s"
    assert format_dry_run_summary(estimate).splitlines() == [
        "Dry run total: 7 model calls, ~52 tokens, ~21s",
        "diagnose: 3 calls, ~24 tokens, ~9s",
        "aggregate: 1 call, ~4 tokens, ~3s",
        "synthesize: 3 calls, ~24 tokens, ~9s",
    ]
