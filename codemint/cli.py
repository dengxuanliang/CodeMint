from pathlib import Path

import typer

from codemint.diagnose.pipeline import run_diagnose
from codemint.io.filesystem import artifact_paths_for_run, ensure_run_directory
from codemint.loaders import detect_loader


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
    diagnoses = run_diagnose(tasks, diagnoses_path)
    typer.echo(f"Wrote {len(diagnoses)} diagnoses to {diagnoses_path}")


@app.command()
def aggregate() -> None:
    """Placeholder command."""


@app.command()
def synthesize() -> None:
    """Placeholder command."""


@app.command()
def run() -> None:
    """Placeholder command."""


def main() -> None:
    app()


if __name__ == "__main__":
    main()
