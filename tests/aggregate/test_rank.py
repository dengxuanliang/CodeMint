from __future__ import annotations

from codemint.models.weakness import CollectiveDiagnosis, WeaknessEntry


def test_rankings_are_parallel_not_composite() -> None:
    from codemint.aggregate.rank import build_rankings

    weaknesses = [
        _weakness(1, "implementation", "off_by_one", frequency=5, trainability=0.4),
        _weakness(2, "modeling", "state_tracking", frequency=2, trainability=0.9),
        _weakness(3, "surface", "syntax", frequency=1, trainability=0.3),
    ]

    rankings = build_rankings(weaknesses)

    assert rankings.by_frequency == [1, 2, 3]
    assert rankings.by_difficulty == [2, 1, 3]
    assert rankings.by_trainability == [2, 1, 3]


def _weakness(rank: int, fault_type: str, tag: str, *, frequency: int, trainability: float) -> WeaknessEntry:
    return WeaknessEntry(
        rank=rank,
        fault_type=fault_type,
        sub_tags=[tag],
        frequency=frequency,
        sample_task_ids=[rank],
        trainability=trainability,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause=f"cause {rank}",
            capability_cliff=f"cliff {rank}",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.8,
        ),
    )
