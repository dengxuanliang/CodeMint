from __future__ import annotations

import json
from pathlib import Path

import pytest

from codemint.config import CodeMintConfig
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


def test_diagnose_applies_token_budget_before_model_analysis(tmp_path: Path) -> None:
    task = TaskRecord(
        task_id=12,
        content="problem " * 10,
        canonical_solution="solution " * 20,
        completion="Wrong answer on hidden case",
        test_code="test " * 30,
        labels={},
        accepted=False,
        metrics={},
        extra={},
    )
    output_path = tmp_path / "diagnoses.jsonl"
    seen: dict[str, str] = {}

    def deep_analyzer(truncated_task: TaskRecord) -> DiagnosisRecord:
        seen["test_code"] = truncated_task.test_code
        seen["canonical_solution"] = truncated_task.canonical_solution
        return _diagnosis(truncated_task.task_id, diagnosis_source="model_deep")

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
        deep_analyzer=deep_analyzer,
        max_input_tokens=25,
    )

    assert [diagnosis.task_id for diagnosis in diagnoses] == [12]
    assert seen["test_code"] == ""
    assert seen["canonical_solution"].split() == ["solution"] * 15


def test_diagnose_logs_model_failures_and_skips_failed_records(tmp_path: Path) -> None:
    tasks = [
        _task(1, "Wrong answer on hidden case"),
        _task(2, "The submission crashes with NameError: helper is not defined"),
    ]
    output_path = tmp_path / "diagnoses.jsonl"

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        raise RuntimeError(f"confirm failed for {task.task_id}")

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        if task.task_id == 1:
            raise RuntimeError("deep failed for 1")
        return _diagnosis(task.task_id, diagnosis_source="model_deep")

    diagnoses = run_diagnose(
        tasks,
        output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert diagnoses == []
    assert output_path.exists()
    assert read_jsonl(output_path) == []
    errors = read_jsonl(tmp_path / "errors.jsonl")
    assert len(errors) == 2
    assert errors[0]["stage"] == "diagnose"
    assert errors[0]["task_id"] == 1
    assert errors[0]["error_type"] == "diagnosis_failed"
    assert errors[1]["task_id"] == 2


def test_diagnose_emits_fine_grained_progress_per_task(tmp_path: Path) -> None:
    tasks = [
        _task(1, "Wrong answer on hidden case"),
        _task(2, "Program returns the wrong total for some inputs."),
    ]
    output_path = tmp_path / "diagnoses.jsonl"
    events: list[dict[str, object]] = []

    diagnoses = run_diagnose(
        tasks,
        output_path,
        rules=build_rules(),
        deep_analyzer=lambda task: _diagnosis(task.task_id, diagnosis_source="model_deep"),
        progress_callback=events.append,
    )

    assert [diagnosis.task_id for diagnosis in diagnoses] == [1, 2]
    assert [event["processed"] for event in events] == [1, 2]
    assert all(event["total"] == 2 for event in events)


def test_diagnose_uses_model_client_when_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task = _task(21, "Wrong answer on hidden case")
    output_path = tmp_path / "diagnoses.jsonl"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return (
                '{"task_id":21,"fault_type":"implementation","sub_tags":["state_tracking"],'
                '"severity":"high","description":"Model-backed diagnosis.","evidence":{"wrong_line":"line",'
                '"correct_approach":"approach","failed_test":"test"},"enriched_labels":{},'
                '"confidence":0.93,"diagnosis_source":"model_deep","prompt_version":"model-v1"}'
            )

    monkeypatch.setattr("codemint.diagnose.pipeline.ModelClient", StubClient)

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
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

    assert diagnoses[0].description == "Model-backed diagnosis."
    assert diagnoses[0].confidence == 0.93


def test_diagnose_normalizes_real_model_style_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task = _task(31, "Wrong answer on hidden case")
    output_path = tmp_path / "diagnoses.jsonl"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return """```json
            {
              "task_id": 31,
              "fault_type": "SyntaxError",
              "sub_tags": ["missing_colon"],
              "severity": "critical",
              "description": "Parser fails immediately.",
              "evidence": "The snippet is missing a colon after the function signature.",
              "enriched_labels": {"location_line": 1},
              "confidence": 0.91,
              "diagnosis_source": "static_analysis",
              "prompt_version": "model-v2"
            }
            ```"""

    monkeypatch.setattr("codemint.diagnose.pipeline.ModelClient", StubClient)

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
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
    assert diagnosis.fault_type == "surface"
    assert diagnosis.severity == "high"
    assert diagnosis.diagnosis_source == "model_deep"
    assert diagnosis.evidence.wrong_line == "The snippet is missing a colon after the function signature."
    assert diagnosis.enriched_labels["location_line"] == "1"


def test_diagnose_normalizes_non_failure_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task = _task(41, "Correct implementation")
    output_path = tmp_path / "diagnoses.jsonl"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return """{
              "task_id": 41,
              "fault_type": "implementation",
              "sub_tags": ["correct_output"],
              "severity": "low",
              "description": "The implementation is correct.",
              "evidence": {"wrong_line": "N/A", "correct_approach": "Already correct.", "failed_test": "N/A"},
              "enriched_labels": {"status": "correct_solution", "test_result": "pass"},
              "confidence": 0.99,
              "diagnosis_source": "correct_solution",
              "prompt_version": "model-v3"
            }"""

    monkeypatch.setattr("codemint.diagnose.pipeline.ModelClient", StubClient)

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
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
    assert diagnosis.diagnosis_source == "non_failure"
    assert diagnosis.is_failure is False


def test_diagnose_normalizes_function_name_mismatch_to_canonical_tag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(42, "def solve_value(x):\n    return x + 1")
    output_path = tmp_path / "diagnoses.jsonl"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return """{
              "task_id": 42,
              "fault_type": "Interface Mismatch",
              "sub_tags": ["public_entry_point_mismatch", "function_name_non_compliance"],
              "severity": "critical",
              "description": "The public function name does not match the test harness contract.",
              "evidence": {
                "completion_snippet": "def solve_value(x):",
                "fix": "def solve(x):",
                "error_message": "NameError: name 'solve' is not defined"
              },
              "enriched_labels": {"root_cause": "function_name_non_compliance"},
              "confidence": 0.98,
              "diagnosis_source": "Static Analysis",
              "prompt_version": "model-v4"
            }"""

    monkeypatch.setattr("codemint.diagnose.pipeline.ModelClient", StubClient)

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
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
    assert diagnosis.fault_type == "surface"
    assert diagnosis.sub_tags == ["function_name_mismatch"]
    assert diagnosis.diagnosis_source == "model_deep"
    assert diagnosis.is_failure is True


def test_diagnose_normalizes_markdown_formatting_to_canonical_tag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(43, "```python\ndef solve(x):\n    return x - 1\n```")
    output_path = tmp_path / "diagnoses.jsonl"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return """{
              "task_id": 43,
              "fault_type": "surface",
              "sub_tags": ["markdown_code_fence", "extraneous_characters"],
              "severity": "low",
              "description": "The solution is wrapped in markdown fences instead of raw code.",
              "evidence": {
                "wrong_line": "```python\\ndef solve(x):\\n    return x - 1\\n```",
                "correct_approach": "Return raw executable Python without markdown formatting.",
                "failed_test": "SyntaxError: invalid syntax"
              },
              "enriched_labels": {"issue_type": "formatting"},
              "confidence": 0.97,
              "diagnosis_source": "deep_analysis",
              "prompt_version": "model-v4"
            }"""

    monkeypatch.setattr("codemint.diagnose.pipeline.ModelClient", StubClient)

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
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
    assert diagnosis.fault_type == "surface"
    assert diagnosis.sub_tags == ["markdown_formatting"]
    assert diagnosis.diagnosis_source == "model_deep"
    assert diagnosis.is_failure is True


def test_diagnose_keeps_missing_code_block_as_failure_even_with_success_like_noise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task(44, "I would solve this by defining a function, but the final code is missing.")
    output_path = tmp_path / "diagnoses.jsonl"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return """{
              "task_id": 44,
              "fault_type": "Missing Output",
              "sub_tags": ["missing_code", "pass"],
              "severity": "high",
              "description": "The response explains the fix but does not provide executable code.",
              "evidence": {
                "wrong_line": "I would solve this by defining a function, but the final code is missing.",
                "correct_approach": "Return only executable code for the required public entry point.",
                "failed_test": "assert solve(1) == 2"
              },
              "enriched_labels": {"status": "correct_solution", "test_result": "pass"},
              "confidence": 0.95,
              "diagnosis_source": "correct_solution",
              "prompt_version": "model-v4"
            }"""

    monkeypatch.setattr("codemint.diagnose.pipeline.ModelClient", StubClient)

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
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
    assert diagnosis.fault_type == "implementation"
    assert diagnosis.sub_tags == ["missing_code_block"]
    assert diagnosis.diagnosis_source == "model_deep"
    assert diagnosis.is_failure is True


@pytest.mark.parametrize(
    ("task_id", "sub_tags"),
    [
        (45, ["correct_output"]),
        (46, ["correct_execution"]),
        (47, ["correct_solution"]),
        (48, ["pass"]),
    ],
)
def test_diagnose_normalizes_success_like_tags_to_non_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task_id: int,
    sub_tags: list[str],
) -> None:
    task = _task(task_id, "Correct implementation")
    output_path = tmp_path / f"diagnoses-{task_id}.jsonl"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return json.dumps(
                {
                    "task_id": task_id,
                    "fault_type": "implementation",
                    "sub_tags": sub_tags,
                    "severity": "low",
                    "description": "The implementation is correct.",
                    "evidence": {
                        "wrong_line": "N/A",
                        "correct_approach": "Already correct.",
                        "failed_test": "N/A",
                    },
                    "enriched_labels": {},
                    "confidence": 0.99,
                    "diagnosis_source": "model_deep",
                    "prompt_version": "model-v4",
                }
            )

    monkeypatch.setattr("codemint.diagnose.pipeline.ModelClient", StubClient)

    diagnoses = run_diagnose(
        [task],
        output_path,
        rules=build_rules(),
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
