from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from codemint.cli import app
from codemint.models.weakness import CausalChain, CollectiveDiagnosis, RankingSet, WeaknessEntry, WeaknessReport


def test_run_synthesize_writes_specs_jsonl(tmp_path: Path) -> None:
    from codemint.synthesize.pipeline import run_synthesize

    output_path = tmp_path / "specs.jsonl"
    report = WeaknessReport(
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

    specs = run_synthesize(
        report,
        output_path,
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
            "key_trap": "Reference the original evidence about lost state transitions.",
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

    assert output_path.exists()
    assert len(specs) == 6
    written = output_path.read_text(encoding="utf-8")
    assert '"spec_id": "spec-0001"' in written
    assert '"prompt_version": "v1"' in written


def test_synthesize_command_reads_weaknesses_and_writes_specs(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "demo-run"
    run_dir.mkdir(parents=True)
    weaknesses_path = run_dir / "weaknesses.json"
    weaknesses_path.write_text(
        WeaknessReport(
            weaknesses=[
                WeaknessEntry(
                    rank=1,
                    fault_type="modeling",
                    sub_tags=["state_tracking"],
                    frequency=4,
                    sample_task_ids=[11, 12],
                    trainability=0.9,
                    collective_diagnosis=CollectiveDiagnosis(
                        refined_root_cause="State invariants break after transitions.",
                        capability_cliff="Long chains require carrying latent state.",
                        misdiagnosed_ids=[],
                        misdiagnosis_corrections={},
                        cluster_coherence=0.93,
                    ),
                )
            ],
            rankings=RankingSet(by_frequency=[1], by_difficulty=[1], by_trainability=[1]),
            causal_chains=[CausalChain(root="state_tracking", downstream=[], training_priority="high")],
            tag_mappings={"state_tracking": "state_tracking"},
        ).model_dump_json(),
        encoding="utf-8",
    )

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
    assert "Wrote 6 synthesized specs" in result.stdout
