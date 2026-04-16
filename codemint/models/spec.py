from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from codemint.models.diagnosis import FaultType


Difficulty = Literal["medium", "hard"]


class TargetWeakness(BaseModel):
    fault_type: FaultType
    sub_tags: list[str]
    root_cause: str
    capability_cliff: str


class ProblemConstraints(BaseModel):
    n_range: list[int]
    value_range: list[int]
    time_limit: str
    memory_limit: str


class ProblemSpec(BaseModel):
    algorithm_type: str
    difficulty: Difficulty
    narrative_theme: str
    constraints: ProblemConstraints
    key_trap: str
    must_cover: list[str]
    must_avoid: list[str]


class VerificationSpec(BaseModel):
    min_test_cases: int
    must_include_edge_cases: list[str]
    brute_force_verifiable: bool
    brute_force_complexity_limit: str


class DiversityTags(BaseModel):
    narrative_theme: str
    data_structure: str
    constraint_scale: str


class GenerationHints(BaseModel):
    solution_approach: str
    common_wrong_approach: str
    distinguishing_test: str


class LanguageConstraint(BaseModel):
    target_languages: list[str]
    language_specific: bool


class SpecRecord(BaseModel):
    spec_id: str
    target_weakness: TargetWeakness
    problem_spec: ProblemSpec
    verification_spec: VerificationSpec
    diversity_tags: DiversityTags
    generation_hints: GenerationHints
    language_constraint: LanguageConstraint
    prompt_version: str
