from __future__ import annotations

import json
from pathlib import Path

from codemint.config import CodeMintConfig
from codemint.io.jsonl import append_jsonl, read_jsonl
from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.spec import SpecRecord
from codemint.models.weakness import WeaknessEntry, WeaknessReport
from codemint.synthesize.allocation import allocate_specs, weakness_key
from codemint.synthesize.diversity import assign_diversity_tags, plan_diversity_tags
from codemint.synthesize.feasibility import check_feasibility
from codemint.synthesize.generate import default_invoke_model, generate_spec


def run_synthesize(
    report: WeaknessReport,
    output_path: Path,
    *,
    config: CodeMintConfig | None = None,
    existing_specs: list[SpecRecord] | None = None,
    invoke_model=None,
    feasibility_check=None,
    diagnoses: list[DiagnosisRecord] | None = None,
    original_evidence: dict[str, dict[str, str]] | None = None,
) -> list[SpecRecord]:
    resolved_config = config or CodeMintConfig()
    synthesize_config = resolved_config.synthesize
    allocated = allocate_specs(report, synthesize_config)
    generated: list[SpecRecord] = []
    prior_specs = list(existing_specs or [])
    model_invoke = invoke_model or default_invoke_model
    evidence_map = original_evidence or build_original_evidence_map(report, diagnoses or [])

    for weakness in sorted(report.weaknesses, key=lambda item: item.rank)[: synthesize_config.top_n]:
        key = weakness_key(weakness)
        count = allocated.get(key, synthesize_config.specs_per_weakness)
        diversity_plan = plan_diversity_tags(
            weakness,
            count,
            synthesize_config,
            prior_specs + generated,
        )
        for slot_index, diversity_tags in enumerate(diversity_plan, start=1):
            spec = _generate_with_regeneration(
                weakness,
                diversity_tags=diversity_tags,
                spec_index=len(generated) + 1,
                difficulty=_difficulty_for_slot(slot_index - 1, count, synthesize_config.difficulty_distribution),
                invoke_model=model_invoke,
                feasibility_check=feasibility_check,
                original_evidence=_evidence_for_weakness(weakness, evidence_map),
                overlap_threshold=synthesize_config.diversity_overlap_threshold,
                existing_specs=prior_specs + generated,
                max_attempts=synthesize_config.max_regeneration_attempts + 1,
            )
            generated.append(spec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    append_jsonl(output_path, [spec.model_dump(mode="json") for spec in generated])
    return generated


def _generate_with_regeneration(
    weakness: WeaknessEntry,
    *,
    diversity_tags,
    spec_index: int,
    difficulty: str,
    invoke_model,
    feasibility_check,
    original_evidence: dict[str, str],
    overlap_threshold: float,
    existing_specs: list[SpecRecord],
    max_attempts: int,
) -> SpecRecord:
    must_avoid_constraints = [f"avoid duplicates of {spec.spec_id}" for spec in existing_specs]

    for _ in range(max_attempts):
        spec = generate_spec(
            weakness,
            diversity_tags=diversity_tags,
            invoke_model=invoke_model,
            original_evidence=original_evidence,
            spec_index=spec_index,
            difficulty=difficulty,
            must_avoid_constraints=must_avoid_constraints,
        )
        diversity_result = assign_diversity_tags(existing_specs, spec.diversity_tags, overlap_threshold)
        feasibility_result = check_feasibility(
            spec,
            original_evidence=original_evidence,
            feasibility_check=feasibility_check,
        )
        if diversity_result.accepted and feasibility_result.accepted:
            return spec

    raise ValueError(f"Failed to synthesize a feasible spec for weakness {weakness_key(weakness)}")


def read_weakness_report(path: Path) -> WeaknessReport:
    return WeaknessReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


def read_diagnoses(path: Path) -> list[DiagnosisRecord]:
    if not path.exists():
        return []
    return [DiagnosisRecord.model_validate(row) for row in read_jsonl(path)]


def build_original_evidence_map(
    report: WeaknessReport,
    diagnoses: list[DiagnosisRecord],
) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[DiagnosisRecord]] = {}
    for diagnosis in diagnoses:
        for sub_tag in diagnosis.sub_tags:
            canonical_tag = report.tag_mappings.get(sub_tag, sub_tag)
            grouped.setdefault(canonical_tag, []).append(diagnosis)

    evidence_map: dict[str, dict[str, str]] = {}
    for weakness in report.weaknesses:
        key = weakness_key(weakness)
        matches = grouped.get(key, [])
        if matches:
            evidence_map[key] = _merge_evidence(matches[:3])

    return evidence_map


def _difficulty_for_slot(slot_index: int, count: int, distribution: str) -> str:
    if distribution == "weighted_hard":
        return "medium" if slot_index % 3 == 1 else "hard"
    return "medium" if slot_index < count / 2 else "hard"


def _evidence_for_weakness(
    weakness: WeaknessEntry,
    original_evidence: dict[str, dict[str, str]] | None,
) -> dict[str, str]:
    key = weakness_key(weakness)
    if original_evidence and key in original_evidence:
        return original_evidence[key]

    return {
        "wrong_line": weakness.collective_diagnosis.refined_root_cause,
        "correct_approach": weakness.collective_diagnosis.capability_cliff,
        "failed_test": f"Reproduce failure around {key}",
    }


def _merge_evidence(diagnoses: list[DiagnosisRecord]) -> dict[str, str]:
    wrong_lines: list[str] = []
    correct_approaches: list[str] = []
    failed_tests: list[str] = []

    for diagnosis in diagnoses:
        if diagnosis.evidence.wrong_line not in wrong_lines:
            wrong_lines.append(diagnosis.evidence.wrong_line)
        if diagnosis.evidence.correct_approach not in correct_approaches:
            correct_approaches.append(diagnosis.evidence.correct_approach)
        if diagnosis.evidence.failed_test not in failed_tests:
            failed_tests.append(diagnosis.evidence.failed_test)

    return {
        "wrong_line": " | ".join(wrong_lines),
        "correct_approach": " | ".join(correct_approaches),
        "failed_test": " | ".join(failed_tests),
    }
