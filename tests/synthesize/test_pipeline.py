from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from codemint.cli import app
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.weakness import CausalChain, CollectiveDiagnosis, RankingSet, WeaknessEntry, WeaknessReport


def test_run_synthesize_uses_real_evidence_from_diagnoses_artifact(tmp_path: Path) -> None:
    from codemint.synthesize.pipeline import build_original_evidence_map, run_synthesize

    output_path = tmp_path / "specs.jsonl"
    report = _report()
    diagnoses = [_diagnosis()]

    specs = run_synthesize(
        report,
        output_path,
        diagnoses=diagnoses,
        invoke_model=lambda prompt: {
            "algorithm_type": "dynamic programming",
            "difficulty": "hard",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 2000],
                "value_range": [0, 10**6],
                "time_limit": "2s",
                "memory_limit": "256MB",
            },
            "key_trap": (
                "Dropping the carry before the final warehouse breaks the last step; "
                "the trap should force preserving carry state through the final transition."
            ),
            "must_cover": ["state_tracking", "transition ordering"],
            "must_avoid": ["greedy"],
            "verification_spec": {
                "min_test_cases": 5,
                "must_include_edge_cases": ["single warehouse"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Track state across stages.",
                "common_wrong_approach": "Collapse states too early.",
                "distinguishing_test": "Optimal answer uses a delayed transition",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        },
    )

    evidence_map = build_original_evidence_map(report, diagnoses)

    assert output_path.exists()
    assert len(specs) == 6
    assert evidence_map["state_tracking"]["wrong_line"] == _diagnosis().evidence.wrong_line
    assert "final warehouse" in specs[0].problem_spec.key_trap.lower()
    written = output_path.read_text(encoding="utf-8")
    assert '"spec_id": "spec-0001"' in written
    assert '"prompt_version": "v1"' in written


def test_synthesize_command_reads_real_evidence_from_diagnoses_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "demo-run"
    run_dir.mkdir(parents=True)
    (run_dir / "weaknesses.json").write_text(_report().model_dump_json(), encoding="utf-8")
    (run_dir / "diagnoses.jsonl").write_text(_diagnosis().model_dump_json() + "\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "synthesize",
            "--output-root",
            str(tmp_path / "artifacts"),
            "--run-id",
            "demo-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (run_dir / "specs.jsonl").exists()
    specs_text = (run_dir / "specs.jsonl").read_text(encoding="utf-8")
    assert "Dropped the carry before processing the final warehouse." in specs_text
    assert "Wrote 6 synthesized specs" in result.stdout


def _report() -> WeaknessReport:
    return WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="modeling",
                sub_tags=["state_tracking"],
                frequency=5,
                sample_task_ids=[1, 2, 3],
                trainability=0.9,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="State invariants break after transitions.",
                    capability_cliff="Long chains require carrying latent state.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.94,
                ),
            )
        ],
        rankings=RankingSet(by_frequency=[1], by_difficulty=[1], by_trainability=[1]),
        causal_chains=[CausalChain(root="state_tracking", downstream=[], training_priority="high")],
        tag_mappings={"state_tracking": "state_tracking"},
    )


def _diagnosis() -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=11,
        fault_type="modeling",
        sub_tags=["state_tracking"],
        severity="high",
        description="State is lost at the final transition.",
        evidence=DiagnosisEvidence(
            wrong_line="Dropped the carry before processing the final warehouse.",
            correct_approach="Preserve carry state through the final transition.",
            failed_test="Final warehouse requires carrying state into the last step.",
        ),
        enriched_labels={},
        confidence=0.9,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )
