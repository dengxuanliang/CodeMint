from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from codemint.config import SynthesizeConfig
from codemint.models.spec import DiversityTags, SpecRecord
from codemint.models.weakness import WeaknessEntry

from codemint.synthesize.allocation import weakness_key
from codemint.synthesize.language_profile import infer_language_profile


CONSTRAINT_SCALES = ("small", "medium", "large")


@dataclass(frozen=True, slots=True)
class DiversityAssignment:
    accepted: bool
    overlap_score: float
    reason: str


def assign_diversity_tags(
    existing_specs: list[SpecRecord],
    candidate_tags: DiversityTags,
    overlap_threshold: float,
) -> DiversityAssignment:
    highest_overlap = 0.0
    for spec in existing_specs:
        overlap = diversity_overlap(spec.diversity_tags, candidate_tags)
        highest_overlap = max(highest_overlap, overlap)
        if overlap > overlap_threshold:
            return DiversityAssignment(
                accepted=False,
                overlap_score=overlap,
                reason=(
                    "Candidate diversity tags overlap above threshold with "
                    f"{spec.spec_id}."
                ),
            )

    return DiversityAssignment(
        accepted=True,
        overlap_score=highest_overlap,
        reason="Candidate diversity tags accepted.",
    )


def diversity_overlap(existing: DiversityTags, candidate: DiversityTags) -> float:
    matches = 0
    if existing.narrative_theme == candidate.narrative_theme:
        matches += 1
    if existing.data_structure == candidate.data_structure:
        matches += 1
    if existing.constraint_scale == candidate.constraint_scale:
        matches += 1
    return matches / 3.0


def plan_diversity_tags(
    weakness: WeaknessEntry,
    count: int,
    config: SynthesizeConfig,
    existing_specs: list[SpecRecord],
) -> list[DiversityTags]:
    themes = _candidate_themes(weakness, count, config)
    planned: list[DiversityTags] = []
    relevant_existing_specs = _relevant_existing_specs(existing_specs, weakness)

    for narrative_theme, data_structure, constraint_scale in product(
        themes,
        config.data_structures,
        CONSTRAINT_SCALES,
    ):
        candidate = DiversityTags(
            narrative_theme=narrative_theme,
            data_structure=data_structure,
            constraint_scale=constraint_scale,
        )
        assignment = assign_diversity_tags(
            relevant_existing_specs + _planned_specs(planned, weakness),
            candidate,
            config.diversity_overlap_threshold,
        )
        if assignment.accepted:
            planned.append(candidate)
            if len(planned) == count:
                return planned

    raise ValueError(f"Unable to assign {count} diversity slots for weakness {weakness_key(weakness)}")


def _planned_specs(planned: list[DiversityTags], weakness: WeaknessEntry) -> list[SpecRecord]:
    language_constraint = _placeholder_language_constraint(weakness)
    return [
        SpecRecord(
            spec_id=f"planned-{index}",
            target_weakness={
                "fault_type": weakness.fault_type,
                "sub_tags": weakness.sub_tags,
                "root_cause": weakness.collective_diagnosis.refined_root_cause,
                "capability_cliff": weakness.collective_diagnosis.capability_cliff,
            },
            problem_spec={
                "algorithm_type": "placeholder",
                "difficulty": "medium",
                "narrative_theme": tags.narrative_theme,
                "constraints": {
                    "n_range": [1, 1],
                    "value_range": [0, 0],
                    "time_limit": "1s",
                    "memory_limit": "64MB",
                },
                "key_trap": "placeholder",
                "must_cover": [],
                "must_avoid": [],
            },
            verification_spec={
                "min_test_cases": 1,
                "must_include_edge_cases": [],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(1)",
            },
            diversity_tags=tags,
            generation_hints={
                "solution_approach": "placeholder",
                "common_wrong_approach": "placeholder",
                "distinguishing_test": "placeholder",
            },
            language_constraint=language_constraint,
            prompt_version="planned",
        )
        for index, tags in enumerate(planned, start=1)
    ]


def _placeholder_language_constraint(
    weakness: WeaknessEntry,
) -> dict[str, object]:
    profile = infer_language_profile(
        {
            "wrong_line": weakness.collective_diagnosis.refined_root_cause,
            "correct_approach": weakness.collective_diagnosis.capability_cliff,
            "failed_test": weakness_key(weakness),
        }
    )
    if profile.primary_language == "unknown":
        return {
            "target_languages": ["python"],
            "language_specific": False,
        }
    return {
        "target_languages": list(profile.target_languages),
        "language_specific": profile.language_specific,
    }


def _relevant_existing_specs(existing_specs: list[SpecRecord], weakness: WeaknessEntry) -> list[SpecRecord]:
    target_key = weakness_key(weakness)
    return [
        spec
        for spec in existing_specs
        if weakness_key(spec.target_weakness) == target_key
    ]


def _candidate_themes(
    weakness: WeaknessEntry,
    count: int,
    config: SynthesizeConfig,
) -> list[str]:
    themes = list(dict.fromkeys(config.narrative_themes.generic))
    primary = weakness_key(weakness)

    if not themes or config.narrative_themes.domain_adaptive:
        for index in range(max(count * 3, 6)):
            suffix = "" if index == 0 else f"_{index + 1}"
            themes.append(f"{primary}{suffix}")

    return list(dict.fromkeys(themes))
