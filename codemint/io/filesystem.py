from __future__ import annotations

from pathlib import Path


def ensure_run_directory(output_root: Path, run_id: str) -> Path:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def artifact_paths_for_run(run_dir: Path) -> dict[str, Path]:
    return {
        "diagnoses": run_dir / "diagnoses.jsonl",
        "weaknesses": run_dir / "weaknesses.json",
        "specs": run_dir / "specs.jsonl",
        "run_metadata": run_dir / "run_metadata.json",
    }
