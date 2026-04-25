from __future__ import annotations

from pathlib import Path

from codemint.config import CodeMintConfig
from codemint.diagnose.item_mode import run_item_mode
from codemint.io.jsonl import read_jsonl
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.custom import build_rules


def test_item_mode_persists_and_returns_expected_diagnoses(tmp_path: Path) -> None:
    tasks = [
        _task(1, "The submission crashes with NameError: helper is not defined"),
        _task(2, "Program returns the wrong total for some inputs."),
    ]
    calls: list[tuple[str, int, str]] = []

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        calls.append(("confirm", task.task_id, rule.rule_id))
        return _diagnosis(
            task.task_id,
            diagnosis_source="rule_confirmed_by_model",
            fault_type=rule.fault_type,
            severity=rule.severity,
            sub_tags=[rule.sub_tag],
            description=f"Confirmed from rule {rule.rule_id}.",
        )

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(("deep", task.task_id, ""))
        return _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            description=f"Deep analysis for task {task.task_id}.",
        )

    output_path = tmp_path / "item.jsonl"
    result = run_item_mode(
        tasks,
        output_path=output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert calls == [("confirm", 1, "R002"), ("deep", 2, "")]
    assert [row.task_id for row in result] == [1, 2]
    assert [row.diagnosis_source for row in result] == ["rule_confirmed_by_model", "model_deep"]
    assert read_jsonl(output_path) == [
        row.model_dump(mode="json") for row in result
    ]


def test_item_mode_preserves_existing_rows_and_skips_completed_tasks(tmp_path: Path) -> None:
    tasks = [
        _task(1, "The submission crashes with NameError: helper is not defined"),
        _task(2, "Program returns the wrong total for some inputs."),
    ]
    existing = _diagnosis(
        1,
        diagnosis_source="rule_only",
        sub_tags=["undefined_variable"],
        severity="medium",
        description="Precomputed diagnosis.",
    )
    output_path = tmp_path / "item.jsonl"
    output_path.write_text(existing.model_dump_json() + "\n", encoding="utf-8")
    calls: list[int] = []

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        calls.append(task.task_id)
        return _diagnosis(task.task_id, diagnosis_source="rule_confirmed_by_model")

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(task.task_id)
        return _diagnosis(task.task_id, diagnosis_source="model_deep")

    result = run_item_mode(
        tasks,
        output_path=output_path,
        rules=build_rules(),
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert [row.task_id for row in result] == [1, 2]
    assert calls == [2]
    assert read_jsonl(output_path)[0] == existing.model_dump(mode="json")
    assert read_jsonl(output_path)[1]["diagnosis_source"] == "model_deep"


def test_item_mode_honors_explicit_empty_rules_list(tmp_path: Path) -> None:
    task = _task(3, "The submission crashes with NameError: helper is not defined")
    output_path = tmp_path / "item.jsonl"
    calls: list[tuple[str, int]] = []

    def confirm_analyzer(task: TaskRecord, rule) -> DiagnosisRecord:
        calls.append(("confirm", task.task_id))
        return _diagnosis(task.task_id, diagnosis_source="rule_confirmed_by_model")

    def deep_analyzer(task: TaskRecord) -> DiagnosisRecord:
        calls.append(("deep", task.task_id))
        return _diagnosis(task.task_id, diagnosis_source="model_deep")

    result = run_item_mode(
        [task],
        output_path=output_path,
        rules=[],
        confirm_analyzer=confirm_analyzer,
        deep_analyzer=deep_analyzer,
    )

    assert calls == [("deep", 3)]
    assert result[0].diagnosis_source == "model_deep"
    assert read_jsonl(output_path) == [result[0].model_dump(mode="json")]


def test_item_mode_model_payload_includes_language_markdown_contract_and_truncation(monkeypatch, tmp_path: Path) -> None:
    task = TaskRecord(
        task_id=55,
        content="Please wrap the answering code with markdown code block format, e.g. ```html [code] ```",
        canonical_solution="solution " * 200,
        completion=("head " * 300) + "```html\n<form></form>\n```" + ("tail " * 300),
        test_code=("setup " * 300) + "#<INSERT>\nexpect(true).toBe(true)\n" + ("check " * 300),
        labels={
            "programming_language": "html",
            "execution_language": "jest",
            "difficulty": "hard",
        },
        accepted=False,
        metrics={},
        extra={},
    )
    seen_prompts: list[str] = []

    class FakeClient:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            seen_prompts.append(user_prompt)
            return (
                '{"task_id": 55, "fault_type": "implementation", "sub_tags": ["non_executable_code"], '
                '"severity": "medium", "description": "diagnosis", '
                '"evidence": {"wrong_line": "line", "correct_approach": "approach", "failed_test": "test"}, '
                '"enriched_labels": {}, "confidence": 0.8, "diagnosis_source": "model_deep", "prompt_version": "v1"}'
            )

    monkeypatch.setattr("codemint.diagnose.item_mode._build_model_client", lambda config: FakeClient())

    run_item_mode(
        [task],
        output_path=tmp_path / "item.jsonl",
        rules=[],
        config=CodeMintConfig.model_validate({"model": {"max_input_tokens": 80, "analysis_model": "fake"}}),
    )

    assert len(seen_prompts) == 1
    prompt = seen_prompts[0]
    assert '"programming_language": "html"' in prompt
    assert '"execution_language": "jest"' in prompt
    assert '"prompt_requests_markdown_wrapper": true' in prompt
    assert '"completion_truncated": true' in prompt
    assert '"test_code_truncated": true' in prompt


def test_item_mode_prioritizes_unrequested_markdown_wrapper_over_model_logic_error(
    tmp_path: Path,
) -> None:
    task = TaskRecord(
        task_id=56,
        content=(
            "Please convert the following code to R, with the function signature "
            "`filter_by_prefix <- function(strings, prefix)`.\n"
            "Only implement the target function."
        ),
        canonical_solution="filter_by_prefix <- function(strings, prefix) strings",
        completion="```R\nfilter_by_prefix <- function(strings, prefix) {\n  strings[startsWith(strings, prefix)]\n}\n```",
        test_code="#<INSERT>\nstopifnot(TRUE)",
        labels={"programming_language": "R", "execution_language": "R"},
        accepted=False,
        metrics={},
        extra={},
    )

    result = run_item_mode(
        [task],
        output_path=tmp_path / "item.jsonl",
        rules=[],
        deep_analyzer=lambda _: _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            fault_type="implementation",
            sub_tags=["logic_error"],
            severity="high",
            description="Model focused on an underlying logic defect.",
        ),
    )

    assert result[0].fault_type == "surface"
    assert result[0].sub_tags == ["markdown_formatting"]


def test_item_mode_preserves_logic_error_when_markdown_wrapper_is_requested(
    tmp_path: Path,
) -> None:
    task = TaskRecord(
        task_id=57,
        content="Please return the solution wrapped in a markdown ```R``` code block.",
        canonical_solution="all_prefixes <- function(string) substring(string, 1, seq_len(nchar(string)))",
        completion="```R\nall_prefixes <- function(string) {\n  substring(string, 1, seq_len(nchar(string)))\n}\n```",
        test_code="#<INSERT>\nstopifnot(TRUE)",
        labels={"programming_language": "R", "execution_language": "R"},
        accepted=False,
        metrics={},
        extra={},
    )

    result = run_item_mode(
        [task],
        output_path=tmp_path / "item.jsonl",
        rules=[],
        deep_analyzer=lambda _: _diagnosis(
            task.task_id,
            diagnosis_source="model_deep",
            fault_type="implementation",
            sub_tags=["logic_error"],
            severity="high",
            description="Underlying logic defect remains primary when fenced output is required.",
        ),
    )

    assert result[0].fault_type == "implementation"
    assert result[0].sub_tags == ["logic_error"]


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
        confidence=0.5,
        diagnosis_source=diagnosis_source,
        prompt_version="test-v1",
    )
