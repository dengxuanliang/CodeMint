from __future__ import annotations

from codemint.config import CodeMintConfig
from codemint.models.weakness import CausalChain, CollectiveDiagnosis, RankingSet, WeaknessEntry, WeaknessReport


def test_root_cause_and_high_trainability_increase_spec_count() -> None:
    from codemint.synthesize.allocation import allocate_specs

    report = WeaknessReport(
        weaknesses=[
            _weakness(rank=1, fault_type="modeling", sub_tags=["state_tracking"], trainability=0.9),
            _weakness(rank=2, fault_type="implementation", sub_tags=["off_by_one"], trainability=0.6),
        ],
        rankings=RankingSet(by_frequency=[1, 2], by_difficulty=[1, 2], by_trainability=[1, 2]),
        causal_chains=[
            CausalChain(root="state_tracking", downstream=["off_by_one"], training_priority="high"),
        ],
        tag_mappings={"state_tracking": "state_tracking", "off_by_one": "off_by_one"},
    )
    config = CodeMintConfig()

    allocation = allocate_specs(report, config.synthesize)

    assert allocation["state_tracking"] == 6
    assert allocation["off_by_one"] == 3


def test_allocate_specs_limits_top_n_by_unique_weakness_key() -> None:
    from codemint.synthesize.allocation import allocate_specs

    report = WeaknessReport(
        weaknesses=[
            _weakness(rank=1, fault_type="implementation", sub_tags=["function_name_mismatch"], trainability=0.6),
            _weakness(rank=2, fault_type="surface", sub_tags=["function_name_mismatch"], trainability=0.3),
            _weakness(rank=3, fault_type="implementation", sub_tags=["logic_error"], trainability=0.6),
        ],
        rankings=RankingSet(by_frequency=[1, 2, 3], by_difficulty=[1, 2, 3], by_trainability=[1, 2, 3]),
        causal_chains=[],
        tag_mappings={
            "function_name_mismatch": "function_name_mismatch",
            "logic_error": "logic_error",
        },
    )
    config = CodeMintConfig.model_validate({"synthesize": {"top_n": 2}})

    allocation = allocate_specs(report, config.synthesize)

    assert set(allocation) == {"function_name_mismatch", "logic_error"}


def _weakness(
    *,
    rank: int,
    fault_type: str,
    sub_tags: list[str],
    trainability: float,
) -> WeaknessEntry:
    return WeaknessEntry(
        rank=rank,
        fault_type=fault_type,
        sub_tags=sub_tags,
        frequency=3,
        sample_task_ids=[rank],
        trainability=trainability,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause=f"{sub_tags[0]} root cause",
            capability_cliff=f"{sub_tags[0]} cliff",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.9,
        ),
    )
