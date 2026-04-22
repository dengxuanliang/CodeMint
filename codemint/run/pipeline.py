from __future__ import annotations

import json
import inspect
from dataclasses import dataclass
from datetime import UTC, datetime
import time
from pathlib import Path
from typing import Callable

from codemint.aggregate.pipeline import run_aggregate
from codemint.config import CodeMintConfig
from codemint.diagnose.pipeline import run_diagnose
from codemint.io.filesystem import artifact_paths_for_run, ensure_run_directory
from codemint.io.jsonl import read_jsonl
from codemint.loaders import detect_loader
from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.run_metadata import PromptVersions, RunMetadata, RunProgressEvent, RunSummary
from codemint.models.spec import SpecRecord
from codemint.models.weakness import WeaknessReport
from codemint.prompts.registry import load_prompt
from codemint.run.dry_run import RunStage
from codemint.synthesize.allocation import select_top_weaknesses, weakness_key
from codemint.synthesize.allocation import allocate_specs
from codemint.synthesize.pipeline import read_weakness_report, run_synthesize


@dataclass(frozen=True, slots=True)
class RunPipelineResult:
    run_dir: Path
    stages_executed: list[str]
    metadata: RunMetadata


DiagnoseStage = Callable[[list, Path], list[DiagnosisRecord]]
AggregateStage = Callable[[list[DiagnosisRecord], Path], WeaknessReport]
SynthesizeStage = Callable[..., list[SpecRecord]]
ProgressCallback = Callable[[dict], None]


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
    progress_callback: ProgressCallback | None = None,
) -> RunPipelineResult:
    started_at = time.perf_counter()
    resolved_config = config or CodeMintConfig()
    loader = detect_loader(input_paths)
    tasks = loader.load(input_paths)
    run_dir = ensure_run_directory(output_root, run_id)
    artifacts = artifact_paths_for_run(run_dir)
    stages_executed: list[str] = []
    forced_stages = set(_selected_stages(start_from))
    rerun_downstream = False
    existing_specs: list[SpecRecord] = []

    def emit_progress(stage: str, status: str, processed: int, total: int) -> None:
        if progress_callback is None:
            return
        event = RunProgressEvent(
            stage=stage,
            status=status,
            processed=processed,
            total=total,
            errors=_error_count(run_dir / "errors.jsonl"),
            eta_seconds=0 if status == "completed" else _estimate_stage_eta(total, processed),
        )
        progress_callback(event.model_dump(mode="python"))

    diagnoses: list[DiagnosisRecord]
    # Resume contract:
    # - diagnose is the only task-level resumable stage and uses diagnoses.jsonl task_ids
    # - aggregate/synthesize are derived stage outputs and are rerun as whole stages
    if "diagnose" in forced_stages and _should_run_diagnose(artifacts["diagnoses"], len(tasks)):
        emit_progress("diagnose", "started", 0, len(tasks))
        if run_diagnose_stage is not None:
            diagnoses = run_diagnose_stage(tasks, artifacts["diagnoses"])
        else:
            diagnoses = run_diagnose(
                tasks,
                artifacts["diagnoses"],
                config=resolved_config,
                progress_callback=progress_callback,
            )
        emit_progress("diagnose", "completed", min(len(_read_diagnoses(artifacts["diagnoses"])), len(tasks)), len(tasks))
        stages_executed.append("diagnose")
        rerun_downstream = True
    else:
        diagnoses = _read_diagnoses(artifacts["diagnoses"])
        emit_progress("diagnose", "skipped", len(diagnoses), len(tasks))

    should_run_aggregate = "aggregate" in forced_stages and (
        rerun_downstream
        or start_from in {"aggregate", "synthesize"}
        or _should_run_aggregate(artifacts["weaknesses"], diagnoses)
    )
    if should_run_aggregate:
        emit_progress("aggregate", "started", 0, len(diagnoses))
        if run_aggregate_stage is not None:
            report = run_aggregate_stage(diagnoses, artifacts["weaknesses"])
        else:
            report = run_aggregate(
                diagnoses,
                artifacts["weaknesses"],
                config=resolved_config,
                progress_callback=progress_callback,
            )
        emit_progress("aggregate", "completed", len(diagnoses), len(diagnoses))
        stages_executed.append("aggregate")
        rerun_downstream = True
    else:
        report = read_weakness_report(artifacts["weaknesses"])
        emit_progress("aggregate", "skipped", len(diagnoses), len(diagnoses))

    should_run_synthesize = "synthesize" in forced_stages and (
        rerun_downstream
        or start_from == "synthesize"
        or _should_run_synthesize(artifacts["specs"], report)
    )
    if should_run_synthesize:
        total_specs = _planned_synthesize_slot_total(report, resolved_config)
        emit_progress("synthesize", "started", 0, total_specs)
        if artifacts["specs"].exists():
            existing_specs = _read_specs(artifacts["specs"])
        if run_synthesize_stage is not None:
            kwargs = {}
            try:
                parameters = inspect.signature(run_synthesize_stage).parameters
            except (TypeError, ValueError):
                parameters = {}
            accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
            if "diagnoses" in parameters or accepts_kwargs:
                kwargs["diagnoses"] = diagnoses
            if "existing_specs" in parameters or accepts_kwargs:
                kwargs["existing_specs"] = existing_specs
            specs = run_synthesize_stage(report, artifacts["specs"], **kwargs)
        else:
            specs = run_synthesize(
                report,
                artifacts["specs"],
                config=resolved_config,
                diagnoses=diagnoses,
                existing_specs=existing_specs,
                progress_callback=progress_callback,
            )
        completed_total = max(total_specs, len(specs))
        emit_progress("synthesize", "completed", len(specs), completed_total)
        stages_executed.append("synthesize")
    else:
        specs = _read_specs(artifacts["specs"])
        emit_progress("synthesize", "skipped", len(specs), len(specs))

    metadata = RunMetadata(
        run_id=run_id,
        timestamp=datetime.now(UTC),
        config_snapshot=resolved_config.model_dump(mode="json"),
        analysis_model=resolved_config.model.analysis_model or "unknown",
        prompt_versions=_prompt_versions(),
        input_files=[str(path) for path in input_paths],
        input_count=len(tasks),
        stages_executed=stages_executed,
        self_analysis_warning=_same_model_warning(resolved_config),
        summary=_build_summary(
            diagnoses,
            report,
            specs,
            input_count=len(tasks),
            errors_path=run_dir / "errors.jsonl",
            elapsed_seconds=time.perf_counter() - started_at,
            top_n=resolved_config.synthesize.top_n,
        ),
    )
    _write_run_metadata(artifacts["run_metadata"], metadata)
    return RunPipelineResult(run_dir=run_dir, stages_executed=stages_executed, metadata=metadata)


def _should_run_diagnose(diagnoses_path: Path, input_count: int) -> bool:
    if not diagnoses_path.exists():
        return True
    return len(read_jsonl(diagnoses_path)) < input_count


def _read_diagnoses(path: Path) -> list[DiagnosisRecord]:
    return [DiagnosisRecord.model_validate(row) for row in read_jsonl(path)]


def _read_specs(path: Path) -> list[SpecRecord]:
    return [SpecRecord.model_validate(row) for row in read_jsonl(path)]


def _should_run_aggregate(weaknesses_path: Path, diagnoses: list[DiagnosisRecord]) -> bool:
    if not weaknesses_path.exists():
        return True
    report = read_weakness_report(weaknesses_path)
    if not diagnoses:
        return False
    return len(report.weaknesses) == 0


def _should_run_synthesize(specs_path: Path, report: WeaknessReport) -> bool:
    if not specs_path.exists():
        return True
    specs = _read_specs(specs_path)
    if not report.weaknesses:
        return False
    return len(specs) == 0


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
    *,
    input_count: int,
    errors_path: Path,
    elapsed_seconds: float,
    top_n: int,
) -> RunSummary:
    attempted_weaknesses = _attempted_weaknesses(report, top_n)
    covered_weaknesses = _covered_weaknesses(report, specs, attempted_weaknesses)
    return RunSummary(
        diagnosed=len(diagnoses),
        rule_screened=sum(1 for diagnosis in diagnoses if diagnosis.diagnosis_source != "model_deep"),
        model_analyzed=sum(1 for diagnosis in diagnoses if diagnosis.diagnosis_source == "model_deep"),
        non_failures=sum(1 for diagnosis in diagnoses if not diagnosis.is_failure),
        errors=_error_count(errors_path),
        skipped=max(input_count - len(diagnoses), 0),
        elapsed_seconds=elapsed_seconds,
        weaknesses_found=len(report.weaknesses),
        specs_generated=len(specs),
        synthesize_failures=_count_stage_errors(errors_path, "synthesize"),
        specs_by_weakness=_spec_counts_by_weakness(specs),
        synthesize_status=_synthesize_status(report, specs, attempted_weaknesses),
        attempted_weaknesses=attempted_weaknesses,
        covered_weaknesses=covered_weaknesses,
        weaknesses_without_specs=_weaknesses_without_specs(report, specs, attempted_weaknesses),
        synthesize_fallbacks=_count_synthesize_fallbacks(errors_path),
        synthesize_fallbacks_by_weakness=_synthesize_fallbacks_by_weakness(errors_path),
        synthesize_failure_reasons_by_weakness=_synthesize_failure_reasons_by_weakness(errors_path),
    )


def _selected_stages(start_from: RunStage) -> tuple[RunStage, ...]:
    ordered: tuple[RunStage, ...] = ("diagnose", "aggregate", "synthesize")
    return ordered[ordered.index(start_from) :]


def _planned_synthesize_slot_total(report: WeaknessReport, config: CodeMintConfig) -> int:
    selected = select_top_weaknesses(report.weaknesses, config.synthesize.top_n)
    if not selected:
        return 0
    allocated = allocate_specs(report, config.synthesize)
    return sum(allocated.get(weakness_key(weakness), config.synthesize.specs_per_weakness) for weakness in selected)


def _error_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for row in read_jsonl(path) if _is_error_event(row))


def _count_stage_errors(path: Path, stage: str) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for row in read_jsonl(path)
        if row.get("stage") == stage and _is_error_event(row)
    )


def _spec_counts_by_weakness(specs: list[SpecRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in specs:
        primary_tag = spec.target_weakness.sub_tags[0] if spec.target_weakness.sub_tags else "unknown"
        counts[primary_tag] = counts.get(primary_tag, 0) + 1
    return counts


def _attempted_weaknesses(report: WeaknessReport, top_n: int) -> list[str]:
    if not report.weaknesses or top_n <= 0:
        return []
    attempted: list[str] = []
    seen: set[str] = set()
    for weakness in select_top_weaknesses(report.weaknesses, top_n):
        key = weakness_key(weakness)
        if key in seen:
            continue
        seen.add(key)
        attempted.append(key)
    return attempted


def _covered_weaknesses(
    report: WeaknessReport,
    specs: list[SpecRecord],
    attempted_weaknesses: list[str],
) -> list[str]:
    covered = set(_spec_counts_by_weakness(specs))
    return [weakness for weakness in attempted_weaknesses if weakness in covered]


def _weaknesses_without_specs(
    report: WeaknessReport,
    specs: list[SpecRecord],
    attempted_weaknesses: list[str],
) -> list[str]:
    covered = set(_spec_counts_by_weakness(specs))
    attempted = set(attempted_weaknesses)
    missing: list[str] = []
    for weakness in report.weaknesses:
        primary_tag = weakness.sub_tags[0] if weakness.sub_tags else "unknown"
        if primary_tag in attempted and primary_tag not in covered:
            missing.append(primary_tag)
    return missing


def _synthesize_status(
    report: WeaknessReport,
    specs: list[SpecRecord],
    attempted_weaknesses: list[str],
) -> str:
    if not report.weaknesses:
        return "skipped"
    return "success" if not _weaknesses_without_specs(report, specs, attempted_weaknesses) else "degraded"


def _synthesize_failure_reasons_by_weakness(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    reasons: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        if row.get("stage") != "synthesize" or not _is_error_event(row):
            continue
        weakness = str(row.get("weakness", "unknown"))
        reason = str(row.get("message", ""))
        reasons.setdefault(weakness, [])
        if reason and reason not in reasons[weakness]:
            reasons[weakness].append(reason)
    return reasons


def _count_synthesize_fallbacks(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for row in read_jsonl(path)
        if row.get("stage") == "synthesize" and row.get("event_type") == "fallback_used"
    )


def _synthesize_fallbacks_by_weakness(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for row in read_jsonl(path):
        if row.get("stage") != "synthesize" or row.get("event_type") != "fallback_used":
            continue
        weakness = str(row.get("weakness", "unknown"))
        counts[weakness] = counts.get(weakness, 0) + 1
    return counts


def _is_error_event(row: dict) -> bool:
    return row.get("event_type") not in {"fallback_used"}




def _same_model_warning(config: CodeMintConfig) -> bool:
    analysis_model = config.model.analysis_model
    evaluated_model = config.model.evaluated_model
    return bool(analysis_model and evaluated_model and analysis_model == evaluated_model)


def _estimate_stage_eta(total: int, processed: int) -> int | None:
    remaining = max(total - processed, 0)
    return remaining * 3


def _write_run_metadata(path: Path, metadata: RunMetadata) -> None:
    payload = metadata.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
