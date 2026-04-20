from __future__ import annotations

from pathlib import Path

from codemint.config import CodeMintConfig
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
    assert estimate.stage_input_tokens == {"diagnose": 24, "aggregate": 4, "synthesize": 24}
    assert estimate.stage_output_tokens == {"diagnose": 96, "aggregate": 16, "synthesize": 72}
    assert estimate.stage_seconds == {"diagnose": 9, "aggregate": 3, "synthesize": 9}
    assert estimate.estimated_model_calls == 7
    assert estimate.estimated_input_tokens == 52
    assert estimate.estimated_output_tokens == 184
    assert estimate.estimated_seconds == 4
    assert estimate.rule_screened == 0
    assert estimate.summary_line == "Estimated model calls: 7 (diagnose: 3, aggregate: 1, synthesize: 3)"
    assert format_dry_run_summary(estimate).splitlines() == [
        "Estimated model calls: 7 (diagnose: 3, aggregate: 1, synthesize: 3)",
        "Estimated tokens: ~52 input, ~184 output",
        "Rule-screened (no model call): 0/3",
        "Estimated time: ~4s (concurrency=5)",
        "diagnose: 3 calls, ~24 input tokens, ~96 output tokens, ~9s",
        "aggregate: 1 call, ~4 input tokens, ~16 output tokens, ~3s",
        "synthesize: 3 calls, ~24 input tokens, ~72 output tokens, ~9s",
    ]


def test_dry_run_summary_uses_configured_concurrency() -> None:
    input_path = Path("tests/fixtures/input/merged_eval.jsonl")
    estimate = estimate_run(
        [input_path],
        config=CodeMintConfig.model_validate({"model": {"max_concurrency": 8}}),
    )

    assert "concurrency=8" in format_dry_run_summary(estimate)
