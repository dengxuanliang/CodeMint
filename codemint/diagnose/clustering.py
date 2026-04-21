from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from codemint.diagnose.fingerprint import DiagnoseFingerprint


@dataclass(frozen=True, slots=True)
class DiagnoseCluster:
    cluster_id: str
    member_task_ids: list[int]
    representative_task_ids: list[int]
    fingerprint_summary: dict[str, str | None]
    min_similarity: float


def cluster_fingerprints(
    fingerprints: list[DiagnoseFingerprint],
    *,
    threshold: float,
    representatives: int,
) -> list[DiagnoseCluster]:
    clusters: list[list[DiagnoseFingerprint]] = []

    for fingerprint in sorted(fingerprints, key=lambda item: item.task_id):
        matched = _find_matching_cluster(fingerprint, clusters, threshold)
        if matched is None:
            clusters.append([fingerprint])
        else:
            matched.append(fingerprint)

    return [
        _build_cluster(index, members, representatives)
        for index, members in enumerate(clusters, start=1)
    ]


def _find_matching_cluster(
    fingerprint: DiagnoseFingerprint,
    clusters: list[list[DiagnoseFingerprint]],
    threshold: float,
) -> list[DiagnoseFingerprint] | None:
    for cluster in clusters:
        if _is_compatible(fingerprint, cluster[0]) and _min_similarity(fingerprint, cluster) >= threshold:
            return cluster
    return None


def _is_compatible(left: DiagnoseFingerprint, right: DiagnoseFingerprint) -> bool:
    return (
        left.rule_hint == right.rule_hint
        and left.entry_point_hint == right.entry_point_hint
        and left.output_format_hint == right.output_format_hint
        and left.assertion_hint == right.assertion_hint
        and left.syntax_hint == right.syntax_hint
    )


def _min_similarity(
    fingerprint: DiagnoseFingerprint,
    members: list[DiagnoseFingerprint],
) -> float:
    return min(
        _completion_similarity(fingerprint.normalized_completion, member.normalized_completion)
        for member in members
    )


def _completion_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _build_cluster(
    index: int,
    members: list[DiagnoseFingerprint],
    representatives: int,
) -> DiagnoseCluster:
    ordered = sorted(members, key=lambda item: item.task_id)
    representative_count = max(representatives, 1)
    primary = ordered[0]
    min_similarity = _cluster_min_similarity(ordered)

    return DiagnoseCluster(
        cluster_id=f"cluster-{index:04d}",
        member_task_ids=[member.task_id for member in ordered],
        representative_task_ids=[member.task_id for member in ordered[:representative_count]],
        fingerprint_summary={
            "rule_hint": primary.rule_hint,
            "entry_point_hint": primary.entry_point_hint,
            "output_format_hint": primary.output_format_hint,
            "assertion_hint": primary.assertion_hint,
            "syntax_hint": primary.syntax_hint,
        },
        min_similarity=min_similarity,
    )


def _cluster_min_similarity(members: list[DiagnoseFingerprint]) -> float:
    if len(members) < 2:
        return 1.0

    similarities: list[float] = []
    for left_index, left in enumerate(members):
        for right in members[left_index + 1 :]:
            similarities.append(
                _completion_similarity(left.normalized_completion, right.normalized_completion)
            )
    return min(similarities)
