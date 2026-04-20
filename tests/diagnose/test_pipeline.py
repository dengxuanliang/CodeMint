from __future__ import annotations

from pathlib import Path

import pytest

from codemint.config import CodeMintConfig
from codemint.diagnose.item_mode import run_item_mode
from codemint.diagnose.pipeline import run_diagnose
from codemint.io.jsonl import read_jsonl
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.custom import build_rules


def test_rule_match_with_high_severity_calls_confirmation_model(tmp_path: Path) -> None:
    task = _task(
        1,
        "The submission crashes with NameError: helper is not defined",
    )
    output_path = tmp_path / "diagnoses.jsonl"
    calls: list[tuple[str, int, str]] = []

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        calls.append(("confirm", task.task_id, rule.rule_id))
        return _diagnosis(
            task.task_id,
            diagnosis_source="rule_confirmed_by_model",
            fault_type="surface",
            severity="medium",
            description="Confirmed undefined variable issue.",
            sub_tags=["undefined_variable", "confirmed"],
        )

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(("deep", task.task_id, ""))
        return _diagnosis(task.task_id, diagnosis_source="model_deep")

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert [diagnosis.task_id for diagnosis in diagnoses] == [1]
    assert calls == [("confirm", 1, "R002")]
    assert diagnoses[0].diagnosis_source == "rule_confirmed_by_model"
    assert read_jsonl(output_path) == [diagnoses[0].model_dump(mode="json")]


def test_run_diagnose_matches_item_mode_behavior(tmp_path: Path) -> None:
    tasks = [
        _task(21, "The submission crashes with NameError: helper is not defined"),
        _task(22, "Program returns the wrong total for some inputs."),
    ]

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        return _diagnosis(
            task.task_id,
            diagnosis_source="rule_confirmed_by_model",
            fault_type=rule.fault_type,
            severity=rule.severity,
            sub_tags=[rule.sub_tag],
            description=f"Confirmed from rule {rule.rule_id}.",
        )

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        return _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            description=f"Deep analysis for task {task.task_id}.",
        )

    pipeline_result = run_diagnose(
        tasks,
        tmp_path / "pipeline.jsonl",
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )
    item_result = run_item_mode(
        tasks,
        output_path=tmp_path / "item.jsonl",
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert [row.model_dump(mode="json") for row in pipeline_result] == [
        row.model_dump(mode="json") for row in item_result
    ]


def test_assertion_error_routes_to_deep_analysis(tmp_path: Path) -> None:
    task = _task(7, "Tests fail with AssertionError: expected 3 == 4")
    output_path = tmp_path / "diagnoses.jsonl"
    calls: list[tuple[str, int]] = []

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        calls.append((f"confirm:{rule.rule_id}", task.task_id))
        return _diagnosis(task.task_id, diagnosis_source="rule_confirmed_by_model")

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(("deep", task.task_id))
        return _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            description="Deep analysis classified the assertion failure.",
        )

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert [diagnosis.task_id for diagnosis in diagnoses] == [7]
    assert calls == [("deep", 7)]
    assert diagnoses[0].diagnosis_source == "model_deep"


def test_existing_diagnosis_rows_are_preserved_and_only_missing_tasks_are_processed(
    tmp_path: Path,
) -> None:
    tasks = [
        _task(1, "NameError: helper is not defined"),
        _task(2, "output format mismatch in the final line"),
    ]
    output_path = tmp_path / "diagnoses.jsonl"
    existing = _diagnosis(
        1,
        diagnosis_source="rule_only",
        description="Precomputed diagnosis.",
    )
    output_path.write_text(existing.model_dump_json() + "\n", encoding="utf-8")
    calls: list[int] = []

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        calls.append(task.task_id)
        return _diagnosis(
            task.task_id,
            diagnosis_source="rule_confirmed_by_model",
            severity="medium",
            description=f"Confirmed from rule {rule.rule_id}.",
        )

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(task.task_id)
        return _diagnosis(task.task_id, diagnosis_source="model_deep")

    diagnoses = run_diagnose(
        tasks,
        output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert [diagnosis.task_id for diagnosis in diagnoses] == [1, 2]
    assert calls == []
    assert [row["task_id"] for row in read_jsonl(output_path)] == [1, 2]
    assert read_jsonl(output_path)[0] == existing.model_dump(mode="json")
    assert read_jsonl(output_path)[1]["diagnosis_source"] == "rule_only"


def test_duplicate_input_task_ids_raise_before_writing(tmp_path: Path) -> None:
    tasks = [
        _task(3, "NameError: helper is not defined"),
        _task(3, "output format mismatch"),
    ]
    output_path = tmp_path / "diagnoses.jsonl"

    with pytest.raises(ValueError, match="Duplicate task_id in input tasks: 3"):
        run_diagnose(tasks, output_path, rules=build_rules())

    assert not output_path.exists()


def test_existing_invalid_diagnosis_row_fails_before_appending(tmp_path: Path) -> None:
    output_path = tmp_path / "diagnoses.jsonl"
    output_path.write_text(
        (
            '{"task_id": 1, "fault_type": "implementation", "sub_tags": ["stub"], '
            '"severity": "low", "description": "existing", "evidence": {"wrong_line": "line", '
            '"correct_approach": "approach", "failed_test": "test"}, "enriched_labels": {}, '
            '"confidence": 0.9, "diagnosis_source": "rule_only"}\n'
        ),
        encoding="utf-8",
    )
    original = output_path.read_text(encoding="utf-8")

    with pytest.raises(Exception):
        run_diagnose([_task(2, "NameError: helper is not defined")], output_path, rules=build_rules())

    assert output_path.read_text(encoding="utf-8") == original


def test_rule_miss_routes_to_deep_analysis(tmp_path: Path) -> None:
    task = _task(9, "Program returns the wrong total for some inputs.")
    output_path = tmp_path / "diagnoses.jsonl"
    calls: list[tuple[str, int]] = []

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        calls.append((f"confirm:{rule.rule_id}", task.task_id))
        return _diagnosis(task.task_id, diagnosis_source="rule_confirmed_by_model")

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(("deep", task.task_id))
        return _diagnosis(task.task_id, diagnosis_source="model_deep")

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert calls == [("deep", 9)]
    assert diagnoses[0].diagnosis_source == "model_deep"


def test_default_confirmation_preserves_rule_metadata(tmp_path: Path) -> None:
    task = _task(11, "The submission crashes with NameError: helper is not defined")
    output_path = tmp_path / "diagnoses.jsonl"

    diagnoses = run_diagnose([task], output_path, rules=build_rules())

    assert diagnoses[0].diagnosis_source == "rule_confirmed_by_model"
    assert diagnoses[0].fault_type == "surface"
    assert diagnoses[0].sub_tags == ["undefined_variable"]
    assert diagnoses[0].severity == "medium"


@pytest.mark.parametrize(
    ("original_tags", "expected_tags"),
    [
        (["public_entry_point_mismatch", "function_name_non_compliance"], ["function_name_mismatch"]),
        (["markdown_code_fence", "extraneous_characters"], ["markdown_formatting"]),
    ],
)
def test_run_diagnose_normalizes_canonical_taxonomy_tags(
    tmp_path: Path,
    original_tags: list[str],
    expected_tags: list[str],
) -> None:
    task = _task(12, "Completion")
    output_path = tmp_path / "diagnoses.jsonl"

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=[],
        deep_analyzer=lambda _: _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            fault_type="implementation",
            sub_tags=original_tags,
        ),
    )

    assert diagnoses[0].sub_tags == expected_tags


def test_run_diagnose_keeps_missing_code_block_as_failure(tmp_path: Path) -> None:
    task = _task(13, "Completion")
    output_path = tmp_path / "diagnoses.jsonl"

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=[],
        deep_analyzer=lambda _: _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            fault_type="implementation",
            sub_tags=["missing_code", "pass"],
            description="Explains the approach but omits executable code.",
        ),
    )

    diagnosis = diagnoses[0]
    assert diagnosis.sub_tags == ["missing_code_block"]
    assert diagnosis.diagnosis_source == "model_deep"
    assert diagnosis.is_failure is True


def test_run_diagnose_uses_model_backed_deep_analyzer_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(15, "The answer explains the fix but does not return code.")
    output_path = tmp_path / "diagnoses.jsonl"
    requests: list[str] = []

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            requests.append(user_prompt)
            return """```json
            {
              "task_id": 15,
              "fault_type": "implementation",
              "sub_tags": ["missing_code_block"],
              "severity": "high",
              "description": "The completion explains the intended fix without returning executable code.",
              "evidence": {
                "wrong_line": "I would solve this by defining solve(x), but the final code block is missing.",
                "correct_approach": "Return executable solve(x) code directly.",
                "failed_test": "assert solve(1) == 2"
              },
              "enriched_labels": {
                "reasoning_mode": "structured"
              },
              "confidence": 0.88,
              "diagnosis_source": "model_deep",
              "prompt_version": "v1"
            }
            ```"""

    monkeypatch.setattr("codemint.diagnose.pipeline.ModelClient", StubClient)

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=[],
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

    diagnosis = diagnoses[0]
    assert requests
    assert diagnosis.sub_tags == ["missing_code_block"]
    assert diagnosis.severity == "high"
    assert diagnosis.description.startswith("The completion explains")
    assert diagnosis.evidence.correct_approach == "Return executable solve(x) code directly."
    assert diagnosis.enriched_labels["reasoning_mode"] == "structured"
    assert diagnosis.prompt_version == "v1"


def test_run_diagnose_retries_after_invalid_model_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(16, "The answer uses solve_value instead of solve.")
    output_path = tmp_path / "diagnoses.jsonl"
    prompts: list[str] = []

    class StubClient:
        def __init__(self, config):
            self.config = config
            self.calls = 0

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            prompts.append(user_prompt)
            self.calls += 1
            if self.calls == 1:
                return '{"task_id":16,"fault_type":"implementation","sub_tags":["function_name_mismatch"]}'
            return """{
              "task_id": 16,
              "fault_type": "implementation",
              "sub_tags": ["function_name_mismatch"],
              "severity": "medium",
              "description": "The public entry point name does not match the harness contract.",
              "evidence": {
                "wrong_line": "def solve_value(x):",
                "correct_approach": "Define the exact solve(x) entry point.",
                "failed_test": "NameError: name 'solve' is not defined"
              },
              "enriched_labels": {},
              "confidence": 0.83,
              "diagnosis_source": "model_deep",
              "prompt_version": "v1"
            }"""

    monkeypatch.setattr("codemint.diagnose.pipeline.ModelClient", StubClient)

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=[],
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

    assert diagnoses[0].sub_tags == ["function_name_mismatch"]
    assert len(prompts) == 2
    assert "Format correction" in prompts[1]


def test_run_diagnose_rejects_non_taxonomy_primary_tag_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(17, "```python\ndef solve(x)\n    return x * 2\n```")
    output_path = tmp_path / "diagnoses.jsonl"
    prompts: list[str] = []

    class StubClient:
        def __init__(self, config):
            self.config = config
            self.calls = 0

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            prompts.append(user_prompt)
            self.calls += 1
            if self.calls == 1:
                return """{
                  "task_id": 17,
                  "fault_type": "implementation",
                  "sub_tags": ["missing_colon"],
                  "severity": "high",
                  "description": "The function definition is malformed.",
                  "evidence": {
                    "wrong_line": "def solve(x)",
                    "correct_approach": "Add the missing colon.",
                    "failed_test": "SyntaxError: invalid syntax"
                  },
                  "enriched_labels": {"flags": "missing_colon"},
                  "confidence": 0.91,
                  "diagnosis_source": "model_deep",
                  "prompt_version": "v1"
                }"""
            return """{
              "task_id": 17,
              "fault_type": "surface",
              "sub_tags": ["markdown_formatting", "syntax_error"],
              "severity": "high",
              "description": "Markdown fences and malformed syntax break execution.",
              "evidence": {
                "wrong_line": "```python\ndef solve(x)\n    return x * 2\n```",
                "correct_approach": "Return raw executable Python code with a valid function definition.",
                "failed_test": "SyntaxError: invalid syntax"
              },
              "enriched_labels": {"flags": "missing_colon"},
              "confidence": 0.93,
              "diagnosis_source": "model_deep",
              "prompt_version": "v1"
            }"""

    monkeypatch.setattr("codemint.diagnose.pipeline.ModelClient", StubClient)

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=[],
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

    assert diagnoses[0].sub_tags[0] in {"syntax_error", "markdown_formatting"}
    assert "missing_colon" not in diagnoses[0].sub_tags
    assert len(prompts) >= 1
    if len(prompts) > 1:
        assert "allowed taxonomy" in prompts[1].lower()


def test_run_diagnose_maps_noncanonical_secondary_tags_back_to_allowed_taxonomy(tmp_path: Path) -> None:
    task = _task(18, "Completion")
    output_path = tmp_path / "diagnoses.jsonl"

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=[],
        deep_analyzer=lambda _: _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            fault_type="surface",
            sub_tags=["missing_code", "missing_colon", "markdown_code_fence"],
        ),
    )

    assert diagnoses[0].sub_tags == ["missing_code_block", "syntax_error", "markdown_formatting"]


@pytest.mark.parametrize("sub_tags", [["correct_output"], ["correct_execution"], ["correct_solution"], ["pass"]])
def test_run_diagnose_normalizes_success_like_tags_to_non_failure(
    tmp_path: Path,
    sub_tags: list[str],
) -> None:
    task = _task(14, "Completion")
    output_path = tmp_path / "diagnoses.jsonl"

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=[],
        deep_analyzer=lambda _: _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            fault_type="implementation",
            sub_tags=sub_tags,
            description="The implementation is correct.",
        ),
    )

    diagnosis = diagnoses[0]
    assert diagnosis.diagnosis_source == "non_failure"
    assert diagnosis.is_failure is False


def _task(task_id: int, completion: str) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        content=f"Task {task_id}",
        canonical_solution="pass",
        completion=completion,
        test_code="assert True",
        labels={},
        accepted=False,
        metrics={},
        extra={},
    )


def _diagnosis(
    task_id: int,
    *,
    diagnosis_source: str,
    fault_type: str = "implementation",
    sub_tags: list[str] | None = None,
    severity: str = "low",
    description: str = "Diagnosis",
) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task_id,
        fault_type=fault_type,
        sub_tags=sub_tags or ["stub"],
        severity=severity,
        description=description,
        evidence=DiagnosisEvidence(
            wrong_line="line",
            correct_approach="approach",
            failed_test="test",
        ),
        enriched_labels={},
        confidence=0.9,
        diagnosis_source=diagnosis_source,
        prompt_version="test-v1",
    )
