from __future__ import annotations

from codemint.models.weakness import RankingSet, WeaknessEntry


_DIFFICULTY_SCORES = {
    "modeling": 1.0,
    "comprehension": 0.9,
    "implementation": 0.8,
    "edge_handling": 0.6,
    "surface": 0.3,
}


def build_rankings(weaknesses: list[WeaknessEntry]) -> RankingSet:
    return RankingSet(
        by_frequency=_rank_ids(weaknesses, key=lambda weakness: (weakness.frequency, weakness.trainability)),
        by_difficulty=_rank_ids(
            weaknesses,
            key=lambda weakness: (_DIFFICULTY_SCORES[weakness.fault_type], weakness.frequency),
        ),
        by_trainability=_rank_ids(weaknesses, key=lambda weakness: (weakness.trainability, weakness.frequency)),
    )


def _rank_ids(weaknesses: list[WeaknessEntry], *, key) -> list[int]:
    ordered = sorted(
        weaknesses,
        key=lambda weakness: (-key(weakness)[0], -key(weakness)[1], weakness.rank),
    )
    return [weakness.rank for weakness in ordered]
