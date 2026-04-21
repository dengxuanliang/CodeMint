from __future__ import annotations

from codemint.diagnose.fingerprint import DiagnoseFingerprint


def test_cluster_tasks_groups_exact_fingerprints_and_selects_representatives() -> None:
    from codemint.diagnose.clustering import cluster_fingerprints

    fingerprints = [
        _fingerprint(2, normalized_completion="def solve(x): return x + 1"),
        _fingerprint(1, normalized_completion="def solve(x): return x + 1"),
        _fingerprint(3, normalized_completion="def solve_value(x): return x + 1", entry_point_hint="solve_value"),
    ]

    clusters = cluster_fingerprints(fingerprints, threshold=0.85, representatives=1)

    assert [cluster.member_task_ids for cluster in clusters] == [[1, 2], [3]]
    assert [cluster.representative_task_ids for cluster in clusters] == [[1], [3]]
    assert clusters[0].fingerprint_summary["entry_point_hint"] == "solve"


def test_cluster_tasks_groups_near_duplicate_completions_above_threshold() -> None:
    from codemint.diagnose.clustering import cluster_fingerprints

    fingerprints = [
        _fingerprint(1, normalized_completion="def solve(x): return x + 1"),
        _fingerprint(2, normalized_completion="def solve(x): return x + 2"),
        _fingerprint(3, normalized_completion="def solve_value(x): return x + 1", entry_point_hint="solve_value"),
    ]

    clusters = cluster_fingerprints(fingerprints, threshold=0.8, representatives=2)

    assert [cluster.member_task_ids for cluster in clusters] == [[1, 2], [3]]
    assert clusters[0].representative_task_ids == [1, 2]
    assert clusters[0].min_similarity >= 0.8


def test_cluster_tasks_keeps_low_similarity_fingerprints_separate() -> None:
    from codemint.diagnose.clustering import cluster_fingerprints

    fingerprints = [
        _fingerprint(1, normalized_completion="def solve(x): return x + 1"),
        _fingerprint(
            2,
            normalized_completion="I would solve this with dynamic programming, but no final code is provided.",
            entry_point_hint=None,
            output_format_hint="prose",
            assertion_hint="assert solve(?) == ?",
        ),
    ]

    clusters = cluster_fingerprints(fingerprints, threshold=0.9, representatives=1)

    assert [cluster.member_task_ids for cluster in clusters] == [[1], [2]]


def _fingerprint(
    task_id: int,
    *,
    normalized_completion: str,
    entry_point_hint: str | None = "solve",
    output_format_hint: str | None = "raw_code",
    assertion_hint: str | None = "assert solve(?) == ?",
) -> DiagnoseFingerprint:
    return DiagnoseFingerprint(
        task_id=task_id,
        rule_hint=None,
        entry_point_hint=entry_point_hint,
        output_format_hint=output_format_hint,
        assertion_hint=assertion_hint,
        syntax_hint=None,
        normalized_completion=normalized_completion,
    )
