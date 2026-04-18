from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from codemint.aggregate.pipeline import run_aggregate
from codemint.config import CodeMintConfig
from codemint.diagnose.pipeline import run_diagnose
from codemint.io.filesystem import artifact_paths_for_run, ensure_run_directory
from codemint.io.jsonl import read_jsonl
from codemint.loaders import detect_loader
from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.run_metadata import PromptVersions, RunMetadata, RunSummary
from codemint.models.spec import SpecRecord
from codemint.models.weakness import WeaknessReport
from codemint.prompts.registry import load_prompt
from codemint.run.dry_run import RunStage
from codemint.synthesize.pipeline import read_weakness_report, run_synthesize


@dataclass(frozen=True, slots=True)
class RunPipelineResult:
    run_dir: Path
    stages_executed: list[str]
    metadata: RunMetadata


DiagnoseStage = Callable[[list, Path], list[DiagnosisRecord]]
AggregateStage = Callable[[list[DiagnosisRecord], Path], WeaknessReport]
SynthesizeStage = Callable[[WeaknessReport, Path], list[SpecRecord]]


def run_pipeline(
    *,
    input_paths: list[Path],
    output_root: Path,
    run_id: str,
    start_from: RunStage = "diagnose",
    config: CodeMintConfig | None = None,
    run_diagnose_stage: DiagnoseStage | None = None,
    run_aggregate_stage: AggregateStage | None = None,
    run_synthesize_stage: SynthesizeStage | None = None,
) -> RunPipelineResult:
    resolved_config = config or CodeMintConfig()
    loader = detect_loader(input_paths)
    tasks = loader.load(input_paths)
    run_dir = ensure_run_directory(output_root, run_id)
    artifacts = artifact_paths_for_run(run_dir)
    stages_executed: list[str] = []
    forced_stages = set(_selected_stages(start_from))
    rerun_downstream = False

    diagnoses: list[DiagnosisRecord]
    if "diagnose" in forced_stages and _should_run_diagnose(artifacts["diagnoses"], len(tasks)):
        diagnoses = (run_diagnose_stage or run_diagnose)(tasks, artifacts["diagnoses"])
        stages_executed.append("diagnose")
        rerun_downstream = True
    else:
        diagnoses = _read_diagnoses(artifacts["diagnoses"])

    should_run_aggregate = "aggregate" in forced_stages and (
        rerun_downstream or start_from in {"aggregate", "synthesize"} or not artifacts["weaknesses"].exists()
    )
    if should_run_aggregate:
        report = (run_aggregate_stage or run_aggregate)(diagnoses, artifacts["weaknesses"])
        stages_executed.append("aggregate")
        rerun_downstream = True
    else:
        report = read_weakness_report(artifacts["weaknesses"])

    should_run_synthesize = "synthesize" in forced_stages and (
        rerun_downstream or start_from == "synthesize" or not artifacts["specs"].exists()
    )
    if should_run_synthesize:
        specs = (run_synthesize_stage or run_synthesize)(report, artifacts["specs"])
        stages_executed.append("synthesize")
    else:
        specs = _read_specs(artifacts["specs"])

    metadata = RunMetadata(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        config_snapshot=resolved_config.model_dump(mode="json"),
        analysis_model=resolved_config.model.analysis_model or "unknown",
        prompt_versions=_prompt_versions(),
        input_files=[str(path) for path in input_paths],
        input_count=len(tasks),
        stages_executed=stages_executed,
        summary=_build_summary(diagnoses, report, specs),
    )
    artifacts["run_metadata"].write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return RunPipelineResult(run_dir=run_dir, stages_executed=stages_executed, metadata=metadata)


def _should_run_diagnose(diagnoses_path: Path, input_count: int) -> bool:
    if not diagnoses_path.exists():
        return True
    return len(read_jsonl(diagnoses_path)) < input_count


def _read_diagnoses(path: Path) -> list[DiagnosisRecord]:
    return [DiagnosisRecord.model_validate(row) for row in read_jsonl(path)]


def _read_specs(path: Path) -> list[SpecRecord]:
    return [SpecRecord.model_validate(row) for row in read_jsonl(path)]


def _prompt_versions() -> PromptVersions:
    return PromptVersions(
        diagnose=load_prompt("diagnose_deep_analysis").version,
        aggregate=load_prompt("aggregate_collective_diagnosis").version,
        synthesize=load_prompt("synthesize_spec_generation").version,
    )


def _build_summary(
    diagnoses: list[DiagnosisRecord],
    report: WeaknessReport,
    specs: list[SpecRecord],
) -> RunSummary:
    return RunSummary(
        diagnosed=len(diagnoses),
        rule_screened=sum(1 for diagnosis in diagnoses if diagnosis.diagnosis_source != "model_deep"),
        model_analyzed=sum(1 for diagnosis in diagnoses if diagnosis.diagnosis_source == "model_deep"),
        errors=0,
        weaknesses_found=len(report.weaknesses),
        specs_generated=len(specs),
    )


def _selected_stages(start_from: RunStage) -> tuple[RunStage, ...]:
    ordered: tuple[RunStage, ...] = ("diagnose", "aggregate", "synthesize")
    return ordered[ordered.index(start_from) :]
