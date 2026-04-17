from __future__ import annotations

from pathlib import Path

import pytest

from codemint.io.filesystem import artifact_paths_for_run, ensure_run_directory


def test_ensure_run_directory_creates_output_run_path(tmp_path: Path) -> None:
    run_dir = ensure_run_directory(tmp_path, "run-123")

    assert run_dir == tmp_path / "run-123"
    assert run_dir.is_dir()


def test_artifact_paths_for_run_returns_expected_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-123"

    paths = artifact_paths_for_run(run_dir)

    assert paths["diagnoses"] == run_dir / "diagnoses.jsonl"
    assert paths["weaknesses"] == run_dir / "weaknesses.json"
    assert paths["specs"] == run_dir / "specs.jsonl"
    assert paths["run_metadata"] == run_dir / "run_metadata.json"


@pytest.mark.parametrize("run_id", ["/tmp/evil", "../evil", "nested/run", r"nested\run"])
def test_ensure_run_directory_rejects_unsafe_run_ids(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        ensure_run_directory(tmp_path, run_id)
