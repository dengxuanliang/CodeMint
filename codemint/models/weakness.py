from __future__ import annotations

from pydantic import BaseModel

from codemint.models.diagnosis import FaultType


class CollectiveDiagnosis(BaseModel):
    refined_root_cause: str
    capability_cliff: str
    misdiagnosed_ids: list[int]
    misdiagnosis_corrections: dict[str, str]
    cluster_coherence: float


class WeaknessEntry(BaseModel):
    rank: int
    fault_type: FaultType
    sub_tags: list[str]
    frequency: int
    sample_task_ids: list[int]
    trainability: float
    collective_diagnosis: CollectiveDiagnosis


class RankingSet(BaseModel):
    by_frequency: list[int]
    by_difficulty: list[int]
    by_trainability: list[int]


class CausalChain(BaseModel):
    root: str
    downstream: list[str]
    training_priority: str


class WeaknessReport(BaseModel):
    weaknesses: list[WeaknessEntry]
    rankings: RankingSet
    causal_chains: list[CausalChain]
    tag_mappings: dict[str, str]
