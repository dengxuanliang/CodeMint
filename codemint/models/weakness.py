from __future__ import annotations

from pydantic import Field

from codemint.models.base import StrictModel
from codemint.models.diagnosis import FaultType


class CollectiveDiagnosis(StrictModel):
    refined_root_cause: str
    capability_cliff: str
    misdiagnosed_ids: list[int]
    misdiagnosis_corrections: dict[str, str]
    cluster_coherence: float = Field(ge=0.0, le=1.0)


class WeaknessEntry(StrictModel):
    rank: int
    fault_type: FaultType
    sub_tags: list[str]
    frequency: int
    sample_task_ids: list[int]
    trainability: float = Field(ge=0.0, le=1.0)
    collective_diagnosis: CollectiveDiagnosis


class RankingSet(StrictModel):
    by_frequency: list[int]
    by_difficulty: list[int]
    by_trainability: list[int]


class CausalChain(StrictModel):
    root: str
    downstream: list[str]
    training_priority: str


class WeaknessReport(StrictModel):
    weaknesses: list[WeaknessEntry]
    rankings: RankingSet
    causal_chains: list[CausalChain]
    tag_mappings: dict[str, str]
