from __future__ import annotations

import json
from pathlib import Path

from codemint.config import CodeMintConfig
from codemint.io.jsonl import append_jsonl
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.spec import (
    DiversityTags,
    GenerationHints,
    LanguageConstraint,
    ProblemConstraints,
    ProblemSpec,
    SpecRecord,
    TargetWeakness,
    VerificationSpec,
)
from codemint.models.weakness import CausalChain, CollectiveDiagnosis, RankingSet, WeaknessEntry, WeaknessReport
from codemint.run.pipeline import run_pipeline


def test_run_metadata_captures_prompt_versions_and_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                '{"task_id": 1, "content": "task one", "canonical_solution": "ok", "completion": "bad one", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
                '{"task_id": 2, "content": "task two", "canonical_solution": "ok", "completion": "bad two", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    report = _report()
    specs = [_spec("spec-0001"), _spec("spec-0002")]

    result = run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="run-001",
        config=CodeMintConfig.model_validate(
            {"model": {"analysis_model": "gpt-4.1-mini", "evaluated_model": "baseline-model"}}
        ),
        run_diagnose_stage=lambda tasks, output_path: _write_diagnoses(output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _write_report(output_path, report),
        run_synthesize_stage=lambda weakness_report, output_path: _write_specs(output_path, specs),
    )

    metadata = json.loads((output_root / "run-001" / "run_metadata.json").read_text(encoding="utf-8"))

    assert result.stages_executed == ["diagnose", "aggregate", "synthesize"]
    assert metadata["run_id"] == "run-001"
    assert metadata["analysis_model"] == "gpt-4.1-mini"
    assert metadata["prompt_versions"] == {
        "diagnose": "v1",
        "aggregate": "v1",
        "synthesize": "v1",
    }
    assert metadata["input_files"] == [str(input_path)]
    assert metadata["input_count"] == 2
    assert metadata["self_analysis_warning"] is False
    assert metadata["summary"] == {
        "diagnosed": 2,
        "rule_screened": 1,
        "model_analyzed": 1,
        "non_failures": 0,
        "errors": 0,
        "skipped": 0,
        "elapsed_seconds": metadata["summary"]["elapsed_seconds"],
        "weaknesses_found": 1,
        "specs_generated": 2,
        "synthesize_failures": 0,
        "specs_by_weakness": {"loop_bound": 2},
        "synthesize_status": "success",
        "attempted_weaknesses": ["loop_bound"],
        "covered_weaknesses": ["loop_bound"],
        "weaknesses_without_specs": [],
        "synthesize_fallbacks": 0,
        "synthesize_fallbacks_by_weakness": {},
        "synthesize_failure_reasons_by_weakness": {},
    }
    assert metadata["summary"]["elapsed_seconds"] >= 0


def test_run_metadata_records_stage_override_execution(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                '{"task_id": 1, "content": "task one", "canonical_solution": "ok", "completion": "bad one", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
                '{"task_id": 2, "content": "task two", "canonical_solution": "ok", "completion": "bad two", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    run_dir = output_root / "run-002"
    run_dir.mkdir(parents=True)
    append_jsonl(
        run_dir / "diagnoses.jsonl",
        [diagnosis.model_dump(mode="json") for diagnosis in _write_diagnoses(run_dir / "diagnoses.jsonl")],
    )
    (run_dir / "weaknesses.json").write_text(_report().model_dump_json(), encoding="utf-8")
    append_jsonl(run_dir / "specs.jsonl", [_spec("spec-0001").model_dump(mode="json")])

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="run-002",
        start_from="synthesize",
        config=CodeMintConfig.model_validate({"model": {"analysis_model": "gpt-4.1-mini"}}),
        run_synthesize_stage=lambda weakness_report, output_path: _write_specs(
            output_path, [_spec("spec-0009")]
        ),
    )

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert metadata["stages_executed"] == ["synthesize"]
    assert metadata["self_analysis_warning"] is False


def test_run_metadata_tracks_non_failures_and_synthesize_stage_errors(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        '{"task_id": 1, "content": "task one", "canonical_solution": "ok", "completion": "ok", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}\n',
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    run_dir = output_root / "run-003"
    run_dir.mkdir(parents=True)
    append_jsonl(
        run_dir / "errors.jsonl",
        [
            {"stage": "synthesize", "weakness": "loop_bound", "error_type": "spec_generation_failed", "message": "boom"},
            {"stage": "aggregate", "task_id": 1, "error_type": "noop", "message": "note"},
        ],
    )

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="run-003",
        config=CodeMintConfig.model_validate({"model": {"analysis_model": "gpt-4.1-mini"}}),
        run_diagnose_stage=lambda tasks, output_path: _write_non_failure_diagnoses(output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _write_report(output_path, _report()),
        run_synthesize_stage=lambda weakness_report, output_path: _write_specs(output_path, [_spec("spec-0010")]),
    )

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["summary"]["non_failures"] == 1
    assert metadata["summary"]["synthesize_failures"] == 1
    assert metadata["summary"]["specs_by_weakness"] == {"loop_bound": 1}
    assert metadata["summary"]["synthesize_status"] == "success"
    assert metadata["summary"]["attempted_weaknesses"] == ["loop_bound"]
    assert metadata["summary"]["covered_weaknesses"] == ["loop_bound"]
    assert metadata["summary"]["weaknesses_without_specs"] == []
    assert metadata["summary"]["synthesize_fallbacks"] == 0
    assert metadata["summary"]["synthesize_fallbacks_by_weakness"] == {}
    assert metadata["summary"]["synthesize_failure_reasons_by_weakness"] == {
        "loop_bound": ["boom"]
    }


def test_run_metadata_marks_synthesize_degraded_when_weaknesses_lack_specs(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                '{"task_id": 1, "content": "task one", "canonical_solution": "ok", "completion": "bad one", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
                '{"task_id": 2, "content": "task two", "canonical_solution": "ok", "completion": "bad two", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="run-004",
        config=CodeMintConfig.model_validate({"model": {"analysis_model": "gpt-4.1-mini"}}),
        run_diagnose_stage=lambda tasks, output_path: _write_diagnoses(output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _write_two_weakness_report(output_path),
        run_synthesize_stage=lambda weakness_report, output_path: _write_specs(output_path, [_spec("spec-0011")]),
    )

    metadata = json.loads((output_root / "run-004" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["summary"]["synthesize_status"] == "degraded"
    assert metadata["summary"]["attempted_weaknesses"] == ["loop_bound", "missing_code"]
    assert metadata["summary"]["covered_weaknesses"] == ["loop_bound"]
    assert metadata["summary"]["weaknesses_without_specs"] == ["missing_code"]


def test_run_metadata_attempted_weaknesses_follow_top_n(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                '{"task_id": 1, "content": "task one", "canonical_solution": "ok", "completion": "bad one", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
                '{"task_id": 2, "content": "task two", "canonical_solution": "ok", "completion": "bad two", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="run-005",
        config=CodeMintConfig.model_validate(
            {
                "model": {"analysis_model": "gpt-4.1-mini"},
                "synthesize": {
                    "specs_per_weakness": 1,
                    "max_per_weakness": 1,
                    "top_n": 1,
                    "narrative_themes": {"generic": ["warehouses"], "domain_adaptive": False},
                    "data_structures": ["array"],
                },
            }
        ),
        run_diagnose_stage=lambda tasks, output_path: _write_diagnoses(output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _write_two_weakness_report(output_path),
        run_synthesize_stage=lambda weakness_report, output_path: _write_specs(output_path, [_spec("spec-0012")]),
    )

    metadata = json.loads((output_root / "run-005" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["summary"]["attempted_weaknesses"] == ["loop_bound"]


def test_run_metadata_missing_and_covered_weaknesses_are_limited_to_attempted_top_n(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                '{"task_id": 1, "content": "task one", "canonical_solution": "ok", "completion": "bad one", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
                '{"task_id": 2, "content": "task two", "canonical_solution": "ok", "completion": "bad two", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="run-006",
        config=CodeMintConfig.model_validate(
            {
                "model": {"analysis_model": "gpt-4.1-mini"},
                "synthesize": {
                    "specs_per_weakness": 1,
                    "max_per_weakness": 1,
                    "top_n": 2,
                    "narrative_themes": {"generic": ["warehouses"], "domain_adaptive": False},
                    "data_structures": ["array"],
                },
            }
        ),
        run_diagnose_stage=lambda tasks, output_path: _write_diagnoses(output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _write_three_weakness_report(output_path),
        run_synthesize_stage=lambda weakness_report, output_path: _write_specs(output_path, [_spec("spec-0013")]),
    )

    metadata = json.loads((output_root / "run-006" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["summary"]["attempted_weaknesses"] == ["loop_bound", "missing_code"]
    assert metadata["summary"]["covered_weaknesses"] == ["loop_bound"]
    assert metadata["summary"]["weaknesses_without_specs"] == ["missing_code"]


def test_run_metadata_deduplicates_covered_weaknesses_preserving_attempted_order(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                '{"task_id": 1, "content": "task one", "canonical_solution": "ok", "completion": "bad one", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
                '{"task_id": 2, "content": "task two", "canonical_solution": "ok", "completion": "bad two", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="run-008",
        config=CodeMintConfig.model_validate({"model": {"analysis_model": "gpt-4.1-mini"}}),
        run_diagnose_stage=lambda tasks, output_path: _write_diagnoses(output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _write_duplicate_covered_weakness_report(output_path),
        run_synthesize_stage=lambda weakness_report, output_path: _write_specs(
            output_path,
            [
                _spec("spec-0015", weakness="missing_code_block"),
                _spec("spec-0016", weakness="function_name_mismatch"),
                _spec("spec-0017", weakness="function_name_mismatch"),
            ],
        ),
    )

    metadata = json.loads((output_root / "run-008" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["summary"]["attempted_weaknesses"] == ["function_name_mismatch", "missing_code_block"]
    assert metadata["summary"]["covered_weaknesses"] == ["function_name_mismatch", "missing_code_block"]


def test_run_metadata_tracks_successful_synthesize_fallback_usage(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        '{"task_id": 1, "content": "task one", "canonical_solution": "ok", "completion": "bad one", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}\n',
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    run_dir = output_root / "run-009"
    run_dir.mkdir(parents=True)
    append_jsonl(
        run_dir / "errors.jsonl",
        [
            {
                "stage": "synthesize",
                "weakness": "missing_code_block",
                "event_type": "fallback_used",
                "message": "deterministic missing code fallback",
            }
        ],
    )

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="run-009",
        config=CodeMintConfig.model_validate({"model": {"analysis_model": "gpt-4.1-mini"}}),
        run_diagnose_stage=lambda tasks, output_path: _write_diagnoses(output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _write_duplicate_covered_weakness_report(output_path),
        run_synthesize_stage=lambda weakness_report, output_path: _write_specs(
            output_path, [_spec("spec-0018", weakness="missing_code_block")]
        ),
    )

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["summary"]["synthesize_fallbacks"] == 1
    assert metadata["summary"]["synthesize_fallbacks_by_weakness"] == {"missing_code_block": 1}


def test_run_metadata_attempted_weaknesses_deduplicate_canonical_keys_within_top_n(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text(
        "\n".join(
            [
                '{"task_id": 1, "content": "task one", "canonical_solution": "ok", "completion": "bad one", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
                '{"task_id": 2, "content": "task two", "canonical_solution": "ok", "completion": "bad two", "test_code": "assert True", "labels": {}, "accepted": false, "metrics": {}, "extra": {}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="run-007",
        config=CodeMintConfig.model_validate(
            {
                "model": {"analysis_model": "gpt-4.1-mini"},
                "synthesize": {
                    "specs_per_weakness": 1,
                    "max_per_weakness": 1,
                    "top_n": 2,
                    "narrative_themes": {"generic": ["warehouses"], "domain_adaptive": False},
                    "data_structures": ["array"],
                },
            }
        ),
        run_diagnose_stage=lambda tasks, output_path: _write_diagnoses(output_path),
        run_aggregate_stage=lambda diagnoses, output_path: _write_duplicate_key_weakness_report(output_path),
        run_synthesize_stage=lambda weakness_report, output_path: _write_specs(output_path, [_spec("spec-0014")]),
    )

    metadata = json.loads((output_root / "run-007" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["summary"]["attempted_weaknesses"] == ["function_name_mismatch", "logic_error"]


def _write_diagnoses(output_path: Path) -> list[DiagnosisRecord]:
    diagnoses = [
        DiagnosisRecord(
            task_id=1,
            fault_type="implementation",
            sub_tags=["rule"],
            severity="low",
            description="Rule-only diagnosis",
            evidence=DiagnosisEvidence(
                wrong_line="line",
                correct_approach="approach",
                failed_test="test",
            ),
            enriched_labels={},
            confidence=0.9,
            diagnosis_source="rule_only",
            prompt_version="diagnose-test-v7",
        ),
        DiagnosisRecord(
            task_id=2,
            fault_type="modeling",
            sub_tags=["deep"],
            severity="medium",
            description="Deep analysis diagnosis",
            evidence=DiagnosisEvidence(
                wrong_line="line",
                correct_approach="approach",
                failed_test="test",
            ),
            enriched_labels={},
            confidence=0.9,
            diagnosis_source="model_deep",
            prompt_version="diagnose-test-v7",
        ),
    ]
    append_jsonl(output_path, [diagnosis.model_dump(mode="json") for diagnosis in diagnoses])
    return diagnoses


def _write_non_failure_diagnoses(output_path: Path) -> list[DiagnosisRecord]:
    diagnoses = [
        DiagnosisRecord(
            task_id=1,
            fault_type="implementation",
            sub_tags=["correct_output"],
            severity="low",
            description="Correct solution",
            evidence=DiagnosisEvidence(
                wrong_line="N/A",
                correct_approach="Already correct.",
                failed_test="N/A",
            ),
            enriched_labels={"status": "correct_solution", "test_result": "pass"},
            confidence=0.99,
            diagnosis_source="non_failure",
            prompt_version="diagnose-test-v8",
        )
    ]
    append_jsonl(output_path, [diagnosis.model_dump(mode="json") for diagnosis in diagnoses])
    return diagnoses


def _write_report(output_path: Path, report: WeaknessReport) -> WeaknessReport:
    output_path.write_text(report.model_dump_json(), encoding="utf-8")
    return report


def _write_specs(output_path: Path, specs: list[SpecRecord]) -> list[SpecRecord]:
    append_jsonl(output_path, [spec.model_dump(mode="json") for spec in specs])
    return specs


def _report() -> WeaknessReport:
    return WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="implementation",
                sub_tags=["loop_bound"],
                frequency=2,
                sample_task_ids=[1, 2],
                trainability=0.7,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Loop bounds drift.",
                    capability_cliff="Longer inputs expose it.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.95,
                ),
            )
        ],
        rankings=RankingSet(by_frequency=[1], by_difficulty=[1], by_trainability=[1]),
        causal_chains=[CausalChain(root="loop_bound", downstream=[], training_priority="high")],
        tag_mappings={"loop_bound": "loop_bound"},
    )


def _write_two_weakness_report(output_path: Path) -> WeaknessReport:
    report = WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="implementation",
                sub_tags=["loop_bound"],
                frequency=2,
                sample_task_ids=[1, 2],
                trainability=0.7,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Loop bounds drift.",
                    capability_cliff="Longer inputs expose it.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.95,
                ),
            ),
            WeaknessEntry(
                rank=2,
                fault_type="implementation",
                sub_tags=["missing_code"],
                frequency=1,
                sample_task_ids=[3],
                trainability=0.6,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Code block omitted.",
                    capability_cliff="Direct code emission fails.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.9,
                ),
            ),
        ],
        rankings=RankingSet(by_frequency=[1, 2], by_difficulty=[1, 2], by_trainability=[1, 2]),
        causal_chains=[CausalChain(root="loop_bound", downstream=["missing_code"], training_priority="high")],
        tag_mappings={"loop_bound": "loop_bound", "missing_code": "missing_code"},
    )
    return _write_report(output_path, report)


def _write_three_weakness_report(output_path: Path) -> WeaknessReport:
    report = WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="implementation",
                sub_tags=["loop_bound"],
                frequency=2,
                sample_task_ids=[1, 2],
                trainability=0.7,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Loop bounds drift.",
                    capability_cliff="Longer inputs expose it.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.95,
                ),
            ),
            WeaknessEntry(
                rank=2,
                fault_type="implementation",
                sub_tags=["missing_code"],
                frequency=1,
                sample_task_ids=[3],
                trainability=0.6,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Code block omitted.",
                    capability_cliff="Direct code emission fails.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.9,
                ),
            ),
            WeaknessEntry(
                rank=3,
                fault_type="surface",
                sub_tags=["markdown_formatting"],
                frequency=1,
                sample_task_ids=[4],
                trainability=0.3,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Markdown fences pollute raw code output.",
                    capability_cliff="Execution fails when formatting wrappers are preserved.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.88,
                ),
            ),
        ],
        rankings=RankingSet(by_frequency=[1, 2, 3], by_difficulty=[1, 2, 3], by_trainability=[1, 2, 3]),
        causal_chains=[CausalChain(root="loop_bound", downstream=["missing_code", "markdown_formatting"], training_priority="high")],
        tag_mappings={
            "loop_bound": "loop_bound",
            "missing_code": "missing_code",
            "markdown_formatting": "markdown_formatting",
        },
    )
    return _write_report(output_path, report)


def _write_duplicate_key_weakness_report(output_path: Path) -> WeaknessReport:
    report = WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="implementation",
                sub_tags=["function_name_mismatch"],
                frequency=4,
                sample_task_ids=[1],
                trainability=0.6,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Wrong entry point name.",
                    capability_cliff="Harness requires exact function name.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.95,
                ),
            ),
            WeaknessEntry(
                rank=2,
                fault_type="surface",
                sub_tags=["function_name_mismatch"],
                frequency=2,
                sample_task_ids=[2],
                trainability=0.3,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Duplicate canonical weakness with different fault type.",
                    capability_cliff="Still the same canonical weakness key.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.9,
                ),
            ),
            WeaknessEntry(
                rank=3,
                fault_type="implementation",
                sub_tags=["logic_error"],
                frequency=1,
                sample_task_ids=[3],
                trainability=0.6,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Wrong formula.",
                    capability_cliff="Logic diverges on hidden cases.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.9,
                ),
            ),
        ],
        rankings=RankingSet(by_frequency=[1, 2, 3], by_difficulty=[1, 2, 3], by_trainability=[1, 2, 3]),
        causal_chains=[],
        tag_mappings={
            "function_name_mismatch": "function_name_mismatch",
            "logic_error": "logic_error",
        },
    )
    return _write_report(output_path, report)


def _write_duplicate_covered_weakness_report(output_path: Path) -> WeaknessReport:
    report = WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="implementation",
                sub_tags=["function_name_mismatch"],
                frequency=6,
                sample_task_ids=[1, 2, 3],
                trainability=0.9,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Wrong public entry point is exposed.",
                    capability_cliff="Harness requires exact solve().",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.95,
                ),
            ),
            WeaknessEntry(
                rank=2,
                fault_type="surface",
                sub_tags=["function_name_mismatch"],
                frequency=5,
                sample_task_ids=[4, 5],
                trainability=0.7,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Same canonical weakness repeated.",
                    capability_cliff="Alternate labels collapse to same weakness.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.9,
                ),
            ),
            WeaknessEntry(
                rank=3,
                fault_type="implementation",
                sub_tags=["missing_code_block"],
                frequency=4,
                sample_task_ids=[6],
                trainability=0.8,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Executable code is omitted.",
                    capability_cliff="Direct code emission fails.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.93,
                ),
            ),
        ],
        rankings=RankingSet(by_frequency=[1, 2, 3], by_difficulty=[1, 2, 3], by_trainability=[1, 3, 2]),
        causal_chains=[
            CausalChain(
                root="function_name_mismatch",
                downstream=["missing_code_block"],
                training_priority="high",
            )
        ],
        tag_mappings={
            "function_name_mismatch": "function_name_mismatch",
            "missing_code_block": "missing_code_block",
        },
    )
    return _write_report(output_path, report)


def _spec(spec_id: str, weakness: str = "loop_bound") -> SpecRecord:
    return SpecRecord(
        spec_id=spec_id,
        target_weakness=TargetWeakness(
            fault_type="implementation",
            sub_tags=[weakness],
            root_cause=f"{weakness} root cause.",
            capability_cliff=f"{weakness} capability cliff.",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="two pointers",
            difficulty="medium",
            narrative_theme="ports",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="bounds",
            must_cover=[weakness],
            must_avoid=["duplicate"],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["n=1"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="ports",
            data_structure="array",
            constraint_scale="medium",
        ),
        generation_hints=GenerationHints(
            solution_approach="Use two pointers.",
            common_wrong_approach="Forget the end condition.",
            distinguishing_test="Length one edge case.",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="synthesize-test-v8",
    )
