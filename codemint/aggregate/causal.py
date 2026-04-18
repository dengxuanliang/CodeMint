from __future__ import annotations

from collections.abc import Callable

from pydantic import Field

from codemint.models.base import StrictModel
from codemint.models.weakness import CausalChain, WeaknessEntry


CausalAnalyzer = Callable[[dict], dict]


class CausalChainResult(StrictModel):
    root: str
    downstream: list[str]
    training_priority: str


class CausalAnalysisResult(StrictModel):
    chains: list[CausalChainResult] = Field(default_factory=list)


def build_causal_chains(
    weaknesses: list[WeaknessEntry],
    analyze: CausalAnalyzer | None = None,
) -> list[CausalChain]:
    if not weaknesses:
        return []

    analyzer = analyze or default_causal_analyze
    result = CausalAnalysisResult.model_validate(
        analyzer(
            {
                "weaknesses": [weakness.model_dump(mode="json") for weakness in weaknesses],
            }
        )
    )
    return [
        CausalChain(
            root=chain.root,
            downstream=chain.downstream,
            training_priority=chain.training_priority,
        )
        for chain in result.chains
    ]


def default_causal_analyze(payload: dict) -> dict:
    weaknesses = payload["weaknesses"]
    ordered = sorted(
        weaknesses,
        key=lambda weakness: (-float(weakness["trainability"]), -int(weakness["frequency"]), int(weakness["rank"])),
    )
    root = ordered[0]["sub_tags"][0]
    downstream = [weakness["sub_tags"][0] for weakness in ordered[1:]]
    return {
        "chains": [
            {
                "root": root,
                "downstream": downstream,
                "training_priority": root,
            }
        ]
    }
