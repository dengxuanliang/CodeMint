from pathlib import Path
from typing import Literal

import typer

from codemint.aggregate.pipeline import run_aggregate
from codemint.config import CodeMintConfig, load_config
from codemint.diagnose.pipeline import run_diagnose
from codemint.io.filesystem import artifact_paths_for_run, ensure_run_directory
from codemint.io.jsonl import read_jsonl
from codemint.loaders import detect_loader
from codemint.logging import format_dry_run_summary, format_progress_event, format_run_summary
from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.run_metadata import RunProgressEvent
from codemint.prompts.registry import set_prompt_override_dir
from codemint.run.dry_run import estimate_run
from codemint.run.pipeline import run_pipeline
from codemint.synthesize.pipeline import read_diagnoses, read_specs, read_weakness_report, run_synthesize


app = typer.Typer(no_args_is_help=True)
RunStage = Literal["diagnose", "aggregate", "synthesize"]


class _ProgressSink:
    def __init__(self) -> None:
        self._last_by_stage: dict[str, tuple[str, int, int, int, int | None]] = {}

    def emit(self, event: dict) -> None:
        parsed = RunProgressEvent.model_validate(event)
        fingerprint = (
            parsed.status,
            parsed.processed,
            parsed.total,
            parsed.errors,
            parsed.eta_seconds,
        )
        if self._last_by_stage.get(parsed.stage) == fingerprint:
            return
        self._last_by_stage[parsed.stage] = fingerprint
        typer.echo(format_progress_event(parsed))


@app.command()
def diagnose(
    input_paths: list[Path] = typer.Argument(..., exists=True, readable=True),
    output_root: Path = typer.Option(Path("artifacts"), "--output-root"),
    run_id: str = typer.Option("latest", "--run-id"),
    config_path: Path | None = typer.Option(None, "--config", exists=True, readable=True),
) -> None:
    """Run the diagnose pipeline."""
    config = CodeMintConfig() if config_path is None else load_config(config_path)
    set_prompt_override_dir(config.prompts.override_dir)
    loader = detect_loader(input_paths)
    tasks = loader.load(input_paths)
    run_dir = ensure_run_directory(output_root, run_id)
    diagnoses_path = artifact_paths_for_run(run_dir)["diagnoses"]
    existing_count = len(read_jsonl(diagnoses_path)) if diagnoses_path.exists() else 0
    diagnoses = run_diagnose(tasks, diagnoses_path, config=config)
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
    input_path: Path | None = typer.Option(None, "-i", "--input", exists=True, readable=True),
    existing_path: Path | None = typer.Option(None, "--existing", exists=True, readable=True),
    output_path: Path | None = typer.Option(None, "-o", "--output"),
    output_root: Path = typer.Option(Path("artifacts"), "--output-root"),
    run_id: str = typer.Option("latest", "--run-id"),
) -> None:
    """Synthesize specs from a weakness report."""
    if input_path is not None or existing_path is not None or output_path is not None:
        if input_path is None or output_path is None:
            raise typer.BadParameter("Incremental mode requires both --input and --output")
        report = read_weakness_report(input_path)
        existing_specs = read_specs(existing_path) if existing_path is not None else None
        specs = run_synthesize(report, output_path, existing_specs=existing_specs)
        typer.echo(f"Wrote {len(specs)} synthesized specs to {output_path}")
        return

    run_dir = ensure_run_directory(output_root, run_id)
    artifacts = artifact_paths_for_run(run_dir)
    report = read_weakness_report(artifacts["weaknesses"])
    diagnoses = read_diagnoses(artifacts["diagnoses"])
    specs = run_synthesize(report, artifacts["specs"], diagnoses=diagnoses)
    typer.echo(f"Wrote {len(specs)} synthesized specs to {artifacts['specs']}")


@app.command()
def run(
    input_paths: list[Path] = typer.Argument(..., exists=True, readable=True),
    output_root: Path = typer.Option(Path("artifacts"), "--output-root"),
    run_id: str = typer.Option("latest", "--run-id"),
    start_from: RunStage = typer.Option("diagnose", "--from"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    config_path: Path | None = typer.Option(None, "--config", exists=True, readable=True),
) -> None:
    """Run the full diagnose -> aggregate -> synthesize pipeline."""
    config = CodeMintConfig() if config_path is None else load_config(config_path)
    set_prompt_override_dir(config.prompts.override_dir)
    if dry_run:
        typer.echo(format_dry_run_summary(estimate_run(input_paths, start_from=start_from, config=config)))
        return

    progress_sink = _ProgressSink()
    result = run_pipeline(
        input_paths=input_paths,
        output_root=output_root,
        run_id=run_id,
        start_from=start_from,
        config=config,
        progress_callback=progress_sink.emit,
    )
    typer.echo(format_run_summary(result.metadata))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
