from __future__ import annotations

import json
from pathlib import Path

from codemint.config import CodeMintConfig
from codemint.diagnose.clustering import DiagnoseCluster
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.task import TaskRecord


def test_propagation_copies_consistent_representative_diagnosis_to_members() -> None:
    from codemint.diagnose.propagation import propagate_cluster_diagnoses

    cluster = _cluster([1, 2, 3], representatives=[1])
    representative = _diagnosis(1, sub_tags=["syntax_error"], confidence=0.9)

    result = propagate_cluster_diagnoses(
        cluster,
        representative_diagnoses=[representative],
        low_confidence_threshold=0.55,
        rediagnose_low_confidence=True,
        max_cluster_size_for_propagation=50,
    )

    assert result.fallback_task_ids == []
    assert [diagnosis.task_id for diagnosis in result.propagated] == [2, 3]
    assert result.propagated[0].sub_tags == ["syntax_error"]
    assert result.propagated[0].enriched_labels["diagnosis_origin"] == "propagated"
    assert result.propagated[0].enriched_labels["cluster_id"] == "cluster-0001"


def test_propagation_falls_back_when_representatives_disagree() -> None:
    from codemint.diagnose.propagation import propagate_cluster_diagnoses

    cluster = _cluster([1, 2, 3], representatives=[1, 2])
    representatives = [
        _diagnosis(1, sub_tags=["syntax_error"], confidence=0.9),
        _diagnosis(2, sub_tags=["markdown_formatting"], confidence=0.9),
    ]

    result = propagate_cluster_diagnoses(
        cluster,
        representative_diagnoses=representatives,
        low_confidence_threshold=0.55,
        rediagnose_low_confidence=True,
        max_cluster_size_for_propagation=50,
    )

    assert result.propagated == []
    assert result.fallback_task_ids == [3]


def test_propagation_falls_back_for_low_confidence_when_enabled() -> None:
    from codemint.diagnose.propagation import propagate_cluster_diagnoses

    cluster = _cluster([1, 2], representatives=[1])
    representative = _diagnosis(1, sub_tags=["missing_code_block"], confidence=0.4)

    result = propagate_cluster_diagnoses(
        cluster,
        representative_diagnoses=[representative],
        low_confidence_threshold=0.55,
        rediagnose_low_confidence=True,
        max_cluster_size_for_propagation=50,
    )

    assert result.propagated == []
    assert result.fallback_task_ids == [2]


def test_run_diagnose_clustered_emits_item_level_records_and_cluster_artifact(tmp_path: Path) -> None:
    from codemint.diagnose.pipeline import run_diagnose

    tasks = [
        _task(1, completion="def solve(x):\n    return x + 1"),
        _task(2, completion="def solve(x):\n    return x + 1"),
        _task(3, completion="def solve_value(x):\n    return x + 1"),
    ]
    config = CodeMintConfig.model_validate(
        {"diagnose": {"processing_mode": "clustered", "cluster_representatives": 1}}
    )
    calls: list[int] = []

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(task.task_id)
        return _diagnosis(task.task_id, sub_tags=["logic_error"], confidence=0.9)

    output_path = tmp_path / "diagnoses.jsonl"

    results = run_diagnose(
        tasks,
        output_path,
        rules=[],
        config=config,
        deep_analyzer=deep_analyzer,
    )

    assert [result.task_id for result in results] == [1, 2, 3]
    assert calls == [1, 3]
    assert results[1].enriched_labels["diagnosis_origin"] == "propagated"
    assert results[1].enriched_labels["cluster_id"] == "cluster-0001"

    clusters = json.loads((tmp_path / "diagnose_clusters.json").read_text(encoding="utf-8"))
    assert [cluster["member_task_ids"] for cluster in clusters["clusters"]] == [[1, 2], [3]]
    assert clusters["summary"]["cluster_count"] == 2


def test_run_diagnose_clustered_uses_rule_hints_to_keep_conflicting_failures_separate(tmp_path: Path) -> None:
    from codemint.diagnose.pipeline import run_diagnose

    tasks = [
        _task(1, completion="def solve(x):\n    return x + 1"),
        _task(2, completion="def solve(x):\n    return x + 1\n# NameError: helper is not defined"),
    ]
    config = CodeMintConfig.model_validate(
        {"diagnose": {"processing_mode": "clustered", "cluster_representatives": 1}}
    )

    results = run_diagnose(
        tasks,
        tmp_path / "diagnoses.jsonl",
        config=config,
        deep_analyzer=lambda task: _diagnosis(task.task_id, sub_tags=["logic_error"], confidence=0.9),
    )

    assert [result.task_id for result in results] == [1, 2]
    clusters = json.loads((tmp_path / "diagnose_clusters.json").read_text(encoding="utf-8"))
    assert [cluster["member_task_ids"] for cluster in clusters["clusters"]] == [[1], [2]]


def _cluster(
    member_task_ids: list[int],
    *,
    representatives: list[int],
) -> DiagnoseCluster:
    return DiagnoseCluster(
        cluster_id="cluster-0001",
        member_task_ids=member_task_ids,
        representative_task_ids=representatives,
        fingerprint_summary={"entry_point_hint": "solve"},
        min_similarity=1.0,
    )


def _diagnosis(
    task_id: int,
    *,
    sub_tags: list[str],
    confidence: float,
) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task_id,
        fault_type="implementation",
        sub_tags=sub_tags,
        severity="medium",
        description="Representative diagnosis.",
        evidence=DiagnosisEvidence(
            wrong_line="def solve(x)",
            correct_approach="Return valid executable code.",
            failed_test="assert solve(1) == 2",
        ),
        enriched_labels={},
        confidence=confidence,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )


def _task(task_id: int, *, completion: str) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        content="Implement solve(x).",
        canonical_solution="def solve(x):\n    return x + 1",
        completion=completion,
        test_code="assert solve(1) == 2",
        labels={},
        accepted=False,
        metrics={},
        extra={},
    )
