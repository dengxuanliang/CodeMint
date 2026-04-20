from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from codemint.models.base import StrictModel
from codemint.models.diagnosis import FaultType


Difficulty = Literal["medium", "hard"]


class TargetWeakness(StrictModel):
    fault_type: FaultType
    sub_tags: list[str]
    root_cause: str
    capability_cliff: str


class ProblemConstraints(StrictModel):
    n_range: list[int]
    value_range: list[int]
    time_limit: str
    memory_limit: str

    @field_validator("n_range", "value_range")
    @classmethod
    def require_two_items(cls, value: list[int]) -> list[int]:
        if len(value) != 2:
            raise ValueError("range must contain exactly 2 integers")
        return value


class ProblemSpec(StrictModel):
    algorithm_type: str
    difficulty: Difficulty
    narrative_theme: str
    constraints: ProblemConstraints
    key_trap: str
    must_cover: list[str]
    must_avoid: list[str]


class VerificationSpec(StrictModel):
    min_test_cases: int = Field(gt=0)
    must_include_edge_cases: list[str]
    brute_force_verifiable: bool
    brute_force_complexity_limit: str


class DiversityTags(StrictModel):
    narrative_theme: str
    data_structure: str
    constraint_scale: str


class GenerationHints(StrictModel):
    solution_approach: str
    common_wrong_approach: str
    distinguishing_test: str


class LanguageConstraint(StrictModel):
    target_languages: list[str]
    language_specific: bool


class SpecRecord(StrictModel):
    spec_id: str
    target_weakness: TargetWeakness
    problem_spec: ProblemSpec
    verification_spec: VerificationSpec
    diversity_tags: DiversityTags
    generation_hints: GenerationHints
    language_constraint: LanguageConstraint
    prompt_version: str
