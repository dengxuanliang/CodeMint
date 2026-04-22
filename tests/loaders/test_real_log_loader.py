from __future__ import annotations

from pathlib import Path

from codemint.loaders.real_log import RealLogFileLoader


def test_real_log_loader_maps_single_file_records_to_task_records() -> None:
    records = RealLogFileLoader().load([Path("tests/fixtures/input/real_log_eval.jsonl")])

    assert len(records) == 1
    assert records[0].task_id == 239
    assert records[0].accepted is False
    assert records[0].metrics == {}
    assert records[0].test_code == "from test_libuv import hello\nassert hello is not None"
    assert records[0].extra == {"model": "glm-5.1"}
    assert "reasoning_content" not in records[0].extra
    assert "pass_at_1" not in records[0].extra
    assert "test_asset" not in records[0].extra
    assert "fewshot" not in records[0].labels
    assert "locale" not in records[0].labels
    assert "is_lctx" not in records[0].labels


def test_real_log_loader_rejects_multiple_files() -> None:
    try:
        RealLogFileLoader().load(
            [
                Path("tests/fixtures/input/real_log_eval.jsonl"),
                Path("tests/fixtures/input/real_log_eval.jsonl"),
            ]
        )
    except ValueError as error:
        assert "expects exactly one input file" in str(error)
    else:
        raise AssertionError("Expected multiple real log files to be rejected")
