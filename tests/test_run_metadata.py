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
    assert metadata["summary"] == {
        "diagnosed": 2,
        "rule_screened": 1,
        "model_analyzed": 1,
        "errors": 0,
        "weaknesses_found": 1,
        "specs_generated": 2,
    }


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


def _spec(spec_id: str) -> SpecRecord:
    return SpecRecord(
        spec_id=spec_id,
        target_weakness=TargetWeakness(
            fault_type="implementation",
            sub_tags=["loop_bound"],
            root_cause="Loop bounds drift.",
            capability_cliff="Longer inputs expose it.",
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
            must_cover=["loop_bound"],
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
