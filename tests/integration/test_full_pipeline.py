from __future__ import annotations

import json
from pathlib import Path

from codemint.aggregate.pipeline import run_aggregate
from codemint.config import CodeMintConfig
from codemint.io.jsonl import append_jsonl, read_jsonl
from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.spec import SpecRecord
from codemint.models.weakness import WeaknessReport
from codemint.run.pipeline import run_pipeline


FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def test_full_pipeline_produces_all_expected_artifacts(tmp_path: Path) -> None:
    input_path = FIXTURES_ROOT / "input" / "merged_eval.jsonl"
    output_root = tmp_path / "artifacts"
    expected_diagnoses = _read_fixture_diagnoses()
    expected_weaknesses = _read_fixture_weaknesses()
    existing_specs = _read_fixture_specs()
    run_dir = output_root / "fixture-run"
    run_dir.mkdir(parents=True)
    append_jsonl(run_dir / "specs.jsonl", [spec.model_dump(mode="json") for spec in existing_specs])

    result = run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="fixture-run",
        config=CodeMintConfig.model_validate(
            {
                "model": {
                    "analysis_model": "gpt-4.1-mini",
                    "evaluated_model": "baseline-model",
                },
                "synthesize": {
                    "specs_per_weakness": 1,
                    "max_per_weakness": 2,
                    "top_n": 1,
                    "diversity_overlap_threshold": 0.8,
                    "narrative_themes": {"generic": ["ports"], "domain_adaptive": False},
                    "data_structures": ["array"],
                },
            }
        ),
        run_diagnose_stage=lambda tasks, output_path: _write_fixture_diagnoses(output_path, expected_diagnoses),
        run_aggregate_stage=lambda diagnoses, output_path: _write_fixture_weaknesses(
            output_path, expected_weaknesses
        ),
    )

    assert result.stages_executed == ["diagnose", "aggregate", "synthesize"]
    assert run_dir.joinpath("diagnoses.jsonl").exists()
    assert run_dir.joinpath("weaknesses.json").exists()
    assert run_dir.joinpath("specs.jsonl").exists()
    assert run_dir.joinpath("run_metadata.json").exists()

    assert [
        DiagnosisRecord.model_validate(row)
        for row in read_jsonl(run_dir / "diagnoses.jsonl")
    ] == expected_diagnoses
    assert WeaknessReport.model_validate_json(
        (run_dir / "weaknesses.json").read_text(encoding="utf-8")
    ) == expected_weaknesses

    specs = [SpecRecord.model_validate(row) for row in read_jsonl(run_dir / "specs.jsonl")]
    assert len(specs) == 2
    assert all(spec.diversity_tags != existing_specs[0].diversity_tags for spec in specs)
    assert "final warehouse" in specs[1].generation_hints.common_wrong_approach
    assert json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))["summary"] == {
        "diagnosed": 2,
        "rule_screened": 1,
        "model_analyzed": 1,
        "errors": 0,
        "weaknesses_found": 1,
        "specs_generated": 2,
    }


def test_parse_failure_is_retried_once_then_logged_to_errors_jsonl(tmp_path: Path) -> None:
    diagnoses = _read_fixture_diagnoses()
    output_path = tmp_path / "weaknesses.json"
    calls = {"count": 0}

    def collective_stub(payload: dict) -> dict:
        calls["count"] += 1
        return {
            "refined_root_cause": "missing capability cliff",
            "cluster_coherence": 0.7,
        }

    report = run_aggregate(diagnoses, output_path, collective_analyze=collective_stub)

    assert calls["count"] == 2
    assert output_path.exists()
    assert report.weaknesses[0].collective_diagnosis.refined_root_cause.startswith("Grouped by")

    errors_path = tmp_path / "errors.jsonl"
    assert errors_path.exists()
    errors = read_jsonl(errors_path)
    assert len(errors) == 1
    assert errors[0]["stage"] == "aggregate"
    assert errors[0]["error_type"] == "collective_parse_failure"
    assert errors[0]["attempts"] == 2
    assert errors[0]["cluster"]["task_ids"] == [101, 102]


def test_same_model_sets_self_analysis_warning(tmp_path: Path) -> None:
    input_path = FIXTURES_ROOT / "input" / "merged_eval.jsonl"
    output_root = tmp_path / "artifacts"

    run_pipeline(
        input_paths=[input_path],
        output_root=output_root,
        run_id="same-model-run",
        config=CodeMintConfig.model_validate(
            {
                "model": {
                    "analysis_model": "gpt-4.1-mini",
                    "evaluated_model": "gpt-4.1-mini",
                }
            }
        ),
        run_diagnose_stage=lambda tasks, output_path: _write_fixture_diagnoses(
            output_path, _read_fixture_diagnoses()
        ),
        run_aggregate_stage=lambda diagnoses, output_path: _write_fixture_weaknesses(
            output_path, _read_fixture_weaknesses()
        ),
    )

    metadata = json.loads(
        (output_root / "same-model-run" / "run_metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["analysis_model"] == "gpt-4.1-mini"
    assert metadata["self_analysis_warning"] is True


def _read_fixture_diagnoses() -> list[DiagnosisRecord]:
    return [
        DiagnosisRecord.model_validate(row)
        for row in read_jsonl(FIXTURES_ROOT / "diagnoses" / "sample_diagnoses.jsonl")
    ]


def _read_fixture_weaknesses() -> WeaknessReport:
    return WeaknessReport.model_validate_json(
        (FIXTURES_ROOT / "aggregate" / "sample_weaknesses.json").read_text(encoding="utf-8")
    )


def _read_fixture_specs() -> list[SpecRecord]:
    return [
        SpecRecord.model_validate(row)
        for row in read_jsonl(FIXTURES_ROOT / "synthesize" / "existing_specs.jsonl")
    ]


def _write_fixture_diagnoses(
    output_path: Path,
    diagnoses: list[DiagnosisRecord],
) -> list[DiagnosisRecord]:
    append_jsonl(output_path, [diagnosis.model_dump(mode="json") for diagnosis in diagnoses])
    return diagnoses


def _write_fixture_weaknesses(
    output_path: Path,
    report: WeaknessReport,
) -> WeaknessReport:
    output_path.write_text(report.model_dump_json(), encoding="utf-8")
    return report
