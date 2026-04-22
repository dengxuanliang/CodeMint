from __future__ import annotations

import json
from pathlib import Path

from scripts.filter_pass_at_1_jsonl import filter_directory, filter_jsonl_file


def test_filter_jsonl_file_keeps_only_numeric_pass_at_1_zero(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": 1, "pass_at_1": 1}),
                json.dumps({"id": 2, "pass_at_1": 0}),
                json.dumps({"id": 3, "pass_at_1": "0"}),
                json.dumps({"id": 4}),
                json.dumps({"id": 5, "pass_at_1": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "filtered.jsonl"

    stats = filter_jsonl_file(input_path, output_path)

    assert stats.input_path == input_path
    assert stats.output_path == output_path
    assert stats.total_rows == 5
    assert stats.kept_rows == 2
    assert [json.loads(line)["id"] for line in output_path.read_text(encoding="utf-8").splitlines()] == [2, 5]


def test_filter_directory_writes_suffix_outputs_without_reprocessing_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "run.jsonl"
    input_path.write_text(
        json.dumps({"id": 1, "pass_at_1": 0}) + "\n" + json.dumps({"id": 2, "pass_at_1": 1}) + "\n",
        encoding="utf-8",
    )
    existing_output = tmp_path / "run.pass_at_1_0.jsonl"
    existing_output.write_text(json.dumps({"id": 99, "pass_at_1": 0}) + "\n", encoding="utf-8")

    stats = filter_directory(tmp_path)

    assert len(stats) == 1
    assert stats[0].input_path == input_path
    assert stats[0].output_path == existing_output
    assert [json.loads(line)["id"] for line in existing_output.read_text(encoding="utf-8").splitlines()] == [1]


def test_filter_jsonl_file_fails_on_invalid_json(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text('{"id": 1, "pass_at_1": 0}\nnot-json\n', encoding="utf-8")

    try:
        filter_jsonl_file(input_path, tmp_path / "out.jsonl")
    except ValueError as error:
        assert "bad.jsonl:2" in str(error)
    else:
        raise AssertionError("Expected invalid JSON to raise ValueError")
