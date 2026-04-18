from __future__ import annotations

from codemint.models.weakness import CollectiveDiagnosis, WeaknessEntry


def test_causal_chain_builds_distinct_root_and_downstream_nodes() -> None:
    from codemint.aggregate.causal import build_causal_chains

    weaknesses = [
        _weakness(1, "implementation", "off_by_one", 3, 0.6),
        _weakness(2, "modeling", "state_tracking", 2, 0.9),
    ]

    def causal_stub(payload: dict) -> dict:
        assert payload["weaknesses"][0]["rank"] == 1
        return {
            "chains": [
                {
                    "root": "state_tracking",
                    "downstream": ["off_by_one"],
                    "training_priority": "state_tracking",
                }
            ]
        }

    chains = build_causal_chains(weaknesses, causal_stub)

    assert len(chains) == 1
    assert chains[0].root == "state_tracking"
    assert chains[0].downstream == ["off_by_one"]
    assert chains[0].training_priority == "state_tracking"


def _weakness(rank: int, fault_type: str, tag: str, frequency: int, trainability: float) -> WeaknessEntry:
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
