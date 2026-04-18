from pathlib import Path

import typer

from codemint.aggregate.pipeline import run_aggregate
from codemint.config import CodeMintConfig
from codemint.diagnose.pipeline import run_diagnose
from codemint.io.filesystem import artifact_paths_for_run, ensure_run_directory
from codemint.io.jsonl import read_jsonl
from codemint.loaders import detect_loader
from codemint.logging import format_dry_run_summary, format_run_summary
from codemint.models.diagnosis import DiagnosisRecord
from codemint.run.dry_run import estimate_run
from codemint.run.pipeline import run_pipeline
from codemint.synthesize.pipeline import read_weakness_report, run_synthesize


app = typer.Typer(no_args_is_help=True)


@app.command()
def diagnose(
    input_paths: list[Path] = typer.Argument(..., exists=True, readable=True),
    output_root: Path = typer.Option(Path("artifacts"), "--output-root"),
    run_id: str = typer.Option("latest", "--run-id"),
) -> None:
    """Run the diagnose pipeline."""
    loader = detect_loader(input_paths)
    tasks = loader.load(input_paths)
    run_dir = ensure_run_directory(output_root, run_id)
    diagnoses_path = artifact_paths_for_run(run_dir)["diagnoses"]
    existing_count = len(read_jsonl(diagnoses_path)) if diagnoses_path.exists() else 0
    diagnoses = run_diagnose(tasks, diagnoses_path)
    typer.echo(
        f"Wrote {len(diagnoses) - existing_count} new diagnoses to {diagnoses_path}"
    )


@app.command()
def aggregate(
    output_root: Path = typer.Option(Path("artifacts"), "--output-root"),
    run_id: str = typer.Option("latest", "--run-id"),
) -> None:
    """Aggregate diagnoses into a weakness report."""
    run_dir = ensure_run_directory(output_root, run_id)
    artifacts = artifact_paths_for_run(run_dir)
    diagnoses_path = artifacts["diagnoses"]
    diagnoses = [DiagnosisRecord.model_validate(row) for row in read_jsonl(diagnoses_path)]
    report = run_aggregate(diagnoses, artifacts["weaknesses"])
    typer.echo(
        f"Wrote weakness report with {len(report.weaknesses)} entries to {artifacts['weaknesses']}"
    )


@app.command()
def synthesize(
    output_root: Path = typer.Option(Path("artifacts"), "--output-root"),
    run_id: str = typer.Option("latest", "--run-id"),
) -> None:
    """Synthesize specs from a weakness report."""
    run_dir = ensure_run_directory(output_root, run_id)
    artifacts = artifact_paths_for_run(run_dir)
    report = read_weakness_report(artifacts["weaknesses"])
    specs = run_synthesize(report, artifacts["specs"])
    typer.echo(f"Wrote {len(specs)} synthesized specs to {artifacts['specs']}")


@app.command()
def run(
    input_paths: list[Path] = typer.Argument(..., exists=True, readable=True),
    output_root: Path = typer.Option(Path("artifacts"), "--output-root"),
    run_id: str = typer.Option("latest", "--run-id"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run the full diagnose -> aggregate -> synthesize pipeline."""
    if dry_run:
        typer.echo(format_dry_run_summary(estimate_run(input_paths)))
        return

    result = run_pipeline(
        input_paths=input_paths,
        output_root=output_root,
        run_id=run_id,
        config=CodeMintConfig(),
    )
    typer.echo(format_run_summary(result.metadata))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
