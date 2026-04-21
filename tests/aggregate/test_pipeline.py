from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from codemint.config import CodeMintConfig
from codemint.cli import app
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord


def test_run_aggregate_repairs_clusters_and_writes_report(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(1, "implementation", ["off_by_one", "loop_bound"], confidence=0.9),
        _diagnosis(2, "implementation", ["off_by_one", "indexing"], confidence=0.8),
    ]
    output_path = tmp_path / "weaknesses.json"
    verify_calls: list[tuple[int, str]] = []

    def verify(record: DiagnosisRecord, level: str):
        verify_calls.append((record.task_id, level))
        return {"level": "cross_model", "status": "passed"}

    report = run_aggregate(
        diagnoses,
        output_path,
        verification_level="cross_model",
        verify=verify,
    )

    assert output_path.exists()
    assert verify_calls == [(1, "cross_model"), (2, "cross_model")]
    assert report.weaknesses[0].fault_type == "implementation"
    assert report.weaknesses[0].sub_tags == ["off_by_one"]
    assert report.weaknesses[0].frequency == 2
    assert report.rankings.by_frequency == [1]
    assert report.rankings.by_difficulty == [1]
    assert report.rankings.by_trainability == [1]
    assert report.tag_mappings == {"off_by_one": "off_by_one"}
    written = output_path.read_text(encoding="utf-8")
    assert '"frequency":2' in written


def test_run_aggregate_default_path_applies_verification_metadata(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    output_path = tmp_path / "weaknesses.json"
    diagnoses = [_diagnosis(9, "implementation", ["off_by_one"])]

    report = run_aggregate(diagnoses, output_path)

    assert report.weaknesses[0].frequency == 1
    assert output_path.exists()
    assert "verification_status=passed" in report.weaknesses[0].collective_diagnosis.refined_root_cause
    assert "verification_level=self_check" in report.weaknesses[0].collective_diagnosis.refined_root_cause
    written = output_path.read_text(encoding="utf-8")
    assert "verification_status=passed" in written
    assert "verification_level=self_check" in written


def test_aggregate_command_reads_diagnoses_and_writes_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "demo-run"
    run_dir.mkdir(parents=True)
    diagnoses_path = run_dir / "diagnoses.jsonl"
    diagnoses_path.write_text(
        "\n".join(
            [
                _diagnosis(1, "implementation", ["off_by_one"]).model_dump_json(),
                _diagnosis(2, "implementation", ["off_by_one"]).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "aggregate",
            "--output-root",
            str(tmp_path / "artifacts"),
            "--run-id",
            "demo-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (run_dir / "weaknesses.json").exists()
    assert "Wrote weakness report" in result.stdout
    written = (run_dir / "weaknesses.json").read_text(encoding="utf-8")
    assert "verification_status=passed" in written
    assert "verification_level=self_check" in written


def test_run_aggregate_emits_fine_grained_progress_per_cluster(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(1, "implementation", ["off_by_one"], confidence=0.9),
        _diagnosis(2, "implementation", ["indexing"], confidence=0.8),
    ]
    output_path = tmp_path / "weaknesses.json"
    events: list[dict[str, object]] = []

    run_aggregate(
        diagnoses,
        output_path,
        progress_callback=events.append,
    )

    assert len(events) >= 2
    assert [event["processed"] for event in events] == [1, 2]
    assert all(event["total"] == 2 for event in events)


def test_run_aggregate_uses_model_client_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [_diagnosis(21, "implementation", ["state_tracking"])]
    output_path = tmp_path / "weaknesses.json"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return (
                '{"refined_root_cause":"Model-backed aggregate diagnosis.","capability_cliff":"'
                'State tracking degrades across similar tasks.","misdiagnosed_ids":[],"misdiagnosis_corrections":{},'
                '"cluster_coherence":0.87,"semantic_merges":[]}'
            )

    monkeypatch.setattr("codemint.aggregate.pipeline.ModelClient", StubClient)

    report = run_aggregate(
        diagnoses,
        output_path,
        config=CodeMintConfig.model_validate(
            {
                "model": {
                    "base_url": "https://example.test",
                    "api_key": "secret",
                    "analysis_model": "gpt-test",
                }
            }
        ),
    )

    assert report.weaknesses[0].collective_diagnosis.refined_root_cause == "Model-backed aggregate diagnosis."
    assert report.weaknesses[0].collective_diagnosis.cluster_coherence == 0.87


def test_run_aggregate_normalizes_real_model_style_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [_diagnosis(31, "implementation", ["off_by_one"])]
    output_path = tmp_path / "weaknesses.json"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return """```json
            {
              "refined_root_cause": "Boundary reasoning breaks on the final transition.",
              "capability_cliff": "Later iterations amplify index mistakes.",
              "misdiagnosed_ids": ["31"],
              "misdiagnosis_corrections": {"31": "implementation:index_bounds"},
              "cluster_coherence": "0.91",
              "semantic_merges": [
                {
                  "source_tag": "index_bounds",
                  "target_tag": "off_by_one",
                  "confirmed": "true"
                }
              ]
            }
            ```"""

    monkeypatch.setattr("codemint.aggregate.pipeline.ModelClient", StubClient)

    report = run_aggregate(
        diagnoses,
        output_path,
        config=CodeMintConfig.model_validate(
            {
                "model": {
                    "base_url": "https://example.test",
                    "api_key": "secret",
                    "analysis_model": "gpt-test",
                }
            }
        ),
    )

    weakness = report.weaknesses[0]
    assert weakness.sub_tags == ["off_by_one"]
    assert weakness.collective_diagnosis.misdiagnosed_ids == [31]
    assert weakness.collective_diagnosis.cluster_coherence == 0.91
    assert report.tag_mappings["index_bounds"] == "off_by_one"


def test_run_aggregate_filters_non_failure_diagnoses(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(41, "implementation", ["correct_output"]),
        _diagnosis(42, "implementation", ["syntax_error"]),
    ]
    diagnoses[0].description = "The solution is correct and passes the test."
    diagnoses[0].evidence = DiagnosisEvidence(
        wrong_line="N/A",
        correct_approach="The provided solution is already correct.",
        failed_test="N/A",
    )
    diagnoses[0].enriched_labels = {"status": "correct_solution", "test_result": "pass"}

    report = run_aggregate(diagnoses, tmp_path / "weaknesses.json")

    assert len(report.weaknesses) == 1
    assert report.weaknesses[0].sub_tags == ["syntax_error"]
    assert report.weaknesses[0].sample_task_ids == [42]


def test_run_aggregate_filters_correct_execution_non_failure(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(51, "implementation", ["correct_execution"]),
        _diagnosis(52, "implementation", ["missing_code_block"]),
    ]
    diagnoses[0].description = "The implementation executes correctly and passes."
    diagnoses[0].enriched_labels = {"functional_correctness": "true", "test_result": "pass"}

    report = run_aggregate(diagnoses, tmp_path / "weaknesses.json")

    assert len(report.weaknesses) == 1
    assert report.weaknesses[0].sub_tags == ["missing_code_block"]


def test_run_aggregate_keeps_function_name_mismatch_distinct_from_other_surface_tags(
    tmp_path: Path,
) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(61, "surface", ["function_name_mismatch"]),
        _diagnosis(62, "surface", ["wrong_function_name"]),
    ]

    def collective_stub(payload: dict) -> dict:
        return {
            "refined_root_cause": "Interface mismatch issue.",
            "capability_cliff": "Callable contract drifts.",
            "misdiagnosed_ids": [],
            "misdiagnosis_corrections": {},
            "cluster_coherence": 0.83,
            "semantic_merges": [
                {
                    "source_tag": "wrong_function_name",
                    "target_tag": "function_name_mismatch",
                    "confirmed": True,
                }
            ],
        }

    report = run_aggregate(
        diagnoses,
        tmp_path / "weaknesses.json",
        collective_analyze=collective_stub,
    )

    assert [entry.sub_tags for entry in report.weaknesses] == [["function_name_mismatch"], ["wrong_function_name"]]
    assert report.tag_mappings["function_name_mismatch"] == "function_name_mismatch"
    assert report.tag_mappings["wrong_function_name"] == "wrong_function_name"


def test_run_aggregate_keeps_markdown_formatting_distinct_from_other_surface_tags(
    tmp_path: Path,
) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(71, "surface", ["markdown_formatting"]),
        _diagnosis(72, "surface", ["extraneous_characters"]),
    ]

    def collective_stub(payload: dict) -> dict:
        return {
            "refined_root_cause": "Formatting issue.",
            "capability_cliff": "Raw output contract drifts.",
            "misdiagnosed_ids": [],
            "misdiagnosis_corrections": {},
            "cluster_coherence": 0.8,
            "semantic_merges": [
                {
                    "source_tag": "extraneous_characters",
                    "target_tag": "markdown_formatting",
                    "confirmed": True,
                }
            ],
        }

    report = run_aggregate(
        diagnoses,
        tmp_path / "weaknesses.json",
        collective_analyze=collective_stub,
    )

    assert [entry.sub_tags for entry in report.weaknesses] == [["extraneous_characters"], ["markdown_formatting"]]
    assert report.tag_mappings["markdown_formatting"] == "markdown_formatting"
    assert report.tag_mappings["extraneous_characters"] == "extraneous_characters"


def test_run_aggregate_keeps_missing_code_block_distinct_from_syntax_error(
    tmp_path: Path,
) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(81, "implementation", ["missing_code_block"]),
        _diagnosis(82, "implementation", ["syntax_error"]),
    ]

    def collective_stub(payload: dict) -> dict:
        return {
            "refined_root_cause": "Non-executable output issue.",
            "capability_cliff": "Response stops being executable.",
            "misdiagnosed_ids": [],
            "misdiagnosis_corrections": {},
            "cluster_coherence": 0.78,
            "semantic_merges": [
                {
                    "source_tag": "missing_code_block",
                    "target_tag": "syntax_error",
                    "confirmed": True,
                }
            ],
        }

    report = run_aggregate(
        diagnoses,
        tmp_path / "weaknesses.json",
        collective_analyze=collective_stub,
    )

    assert [entry.sub_tags for entry in report.weaknesses] == [["missing_code_block"], ["syntax_error"]]
    assert report.tag_mappings["missing_code_block"] == "missing_code_block"
    assert report.tag_mappings["syntax_error"] == "syntax_error"


def test_run_aggregate_excludes_explicit_non_failure_diagnoses(tmp_path: Path) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        _diagnosis(91, "implementation", ["pass"]),
        _diagnosis(92, "implementation", ["syntax_error"]),
    ]
    diagnoses[0].diagnosis_source = "non_failure"

    report = run_aggregate(diagnoses, tmp_path / "weaknesses.json")

    assert len(report.weaknesses) == 1
    assert report.weaknesses[0].sub_tags == ["syntax_error"]
    assert report.weaknesses[0].sample_task_ids == [92]


def test_run_aggregate_preserves_fallback_deep_analysis_placeholders_without_reclassification(
    tmp_path: Path,
) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        DiagnosisRecord(
            task_id=101,
            fault_type="implementation",
            sub_tags=["deep_analysis"],
            severity="medium",
            description="Missing final executable code.",
            evidence=DiagnosisEvidence(
                wrong_line="I would solve this by defining a function and returning x plus one, but the final code block is missing.",
                correct_approach="Return executable solve(x) code directly.",
                failed_test='{"code": "assert solve(1) == 2"}',
                ),
                enriched_labels={"fallback_mode": "true"},
                confidence=0.7,
                diagnosis_source="model_deep",
                prompt_version="deep-v1",
            ),
        DiagnosisRecord(
            task_id=103,
            fault_type="implementation",
            sub_tags=["deep_analysis"],
            severity="medium",
            description="Wrong public entry point.",
            evidence=DiagnosisEvidence(
                wrong_line="```python\ndef solve_value(x):\n    return x * 2\n```",
                correct_approach="Define the exact solve(x) entry point expected by the harness.",
                failed_test="NameError: name 'solve' is not defined",
                ),
                enriched_labels={"fallback_mode": "true"},
                confidence=0.7,
                diagnosis_source="model_deep",
                prompt_version="deep-v1",
            ),
        DiagnosisRecord(
            task_id=104,
            fault_type="implementation",
            sub_tags=["deep_analysis"],
            severity="medium",
            description="Wrong arithmetic result.",
            evidence=DiagnosisEvidence(
                wrong_line="```python\ndef solve(x):\n    return x - 1\n```",
                correct_approach="Return the value required by the assertion instead of a shifted arithmetic shortcut.",
                failed_test='{"code": "assert solve(3) == 2"}',
                ),
                enriched_labels={"fallback_mode": "true"},
                confidence=0.7,
                diagnosis_source="model_deep",
                prompt_version="deep-v1",
            ),
    ]

    report = run_aggregate(diagnoses, tmp_path / "weaknesses.json")

    weakness_tags = [entry.sub_tags[0] for entry in report.weaknesses]
    assert weakness_tags == ["deep_analysis"]


def test_run_aggregate_does_not_reclassify_non_fallback_model_diagnoses(
    tmp_path: Path,
) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        DiagnosisRecord(
            task_id=105,
            fault_type="implementation",
            sub_tags=["deep_analysis"],
            severity="medium",
            description="Model selected an unresolved placeholder tag.",
            evidence=DiagnosisEvidence(
                wrong_line="```python\ndef solve_value(x):\n    return x * 2\n```",
                correct_approach="Return a structured diagnosis from the model.",
                failed_test="NameError: name 'solve' is not defined",
            ),
            enriched_labels={"fallback_mode": "false"},
            confidence=0.82,
            diagnosis_source="model_deep",
            prompt_version="v1",
        )
    ]

    report = run_aggregate(diagnoses, tmp_path / "weaknesses.json")

    assert [entry.sub_tags for entry in report.weaknesses] == [["deep_analysis"]]


def test_run_aggregate_does_not_promote_logic_error_from_secondary_tag_on_function_name_mismatch(
    tmp_path: Path,
) -> None:
    from codemint.aggregate.pipeline import run_aggregate

    diagnoses = [
        DiagnosisRecord(
            task_id=2302,
            fault_type="implementation",
            sub_tags=["function_name_mismatch", "logic_error"],
            severity="high",
            description=(
                "The completion defines a function named 'solver', but the test harness calls 'solve'. "
                "Returning 0 instead of x * 0 is mentioned as a secondary semantic issue."
            ),
            evidence=DiagnosisEvidence(
                wrong_line="def solver(x):\n    return 0",
                correct_approach=(
                    "Define the function with the correct name 'solve' and return the requested expression."
                ),
                failed_test="assert solve(8) == 0",
            ),
            enriched_labels={},
            confidence=0.99,
            diagnosis_source="model_deep",
            prompt_version="v1",
        )
    ]

    def collective_stub(payload: dict) -> dict:
        return {
            "refined_root_cause": "Grouped by implementation/function_name_mismatch",
            "capability_cliff": "Entry-point mismatch is the dominant failure",
            "misdiagnosed_ids": [2302],
            "misdiagnosis_corrections": {"2302": "implementation:logic_error"},
            "cluster_coherence": 0.95,
            "semantic_merges": [],
        }

    report = run_aggregate(
        diagnoses,
        tmp_path / "weaknesses.json",
        collective_analyze=collective_stub,
    )

    weakness_tags = [entry.sub_tags[0] for entry in report.weaknesses]
    assert weakness_tags == ["function_name_mismatch"]


def _diagnosis(
    task_id: int,
    fault_type: str,
    sub_tags: list[str],
    *,
    confidence: float = 0.9,
) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task_id,
        fault_type=fault_type,
        sub_tags=sub_tags,
        severity="medium",
        description=f"Diagnosis {task_id}",
        evidence=DiagnosisEvidence(
            wrong_line="line",
            correct_approach="approach",
            failed_test="test",
        ),
        enriched_labels={},
        confidence=confidence,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )
