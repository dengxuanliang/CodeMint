from __future__ import annotations

from pathlib import Path


def ensure_run_directory(output_root: Path, run_id: str) -> Path:
    if run_id in {"", "."}:
        raise ValueError(f"run_id must name a child directory under output_root, got {run_id!r}")

    run_path = Path(run_id)
    if run_path.is_absolute() or any(part == ".." for part in run_path.parts):
        raise ValueError(f"run_id must be a safe relative directory name, got {run_id!r}")
    if "/" in run_id or "\\" in run_id:
        raise ValueError(f"run_id must not contain path separators, got {run_id!r}")

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
