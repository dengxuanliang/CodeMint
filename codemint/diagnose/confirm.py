from __future__ import annotations

from collections.abc import Callable

from codemint.models.diagnosis import DiagnosisEvidence
from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.builtin import DiagnosisRule


ConfirmAnalyzer = Callable[[TaskRecord, DiagnosisRule], DiagnosisRecord]


def confirm_rule_with_model(
    task: TaskRecord,
    rule: DiagnosisRule,
    analyzer: ConfirmAnalyzer,
) -> DiagnosisRecord:
    return analyzer(task, rule)


def default_confirm_analyzer(task: TaskRecord, rule: DiagnosisRule) -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=task.task_id,
        fault_type=rule.fault_type,
        sub_tags=[rule.sub_tag],
        severity=rule.severity,
        description=f"Model-confirmed diagnosis for {rule.rule_id}.",
        evidence=DiagnosisEvidence(
            wrong_line=task.completion,
            correct_approach=f"Review the matched rule context for {rule.sub_tag} and repair it.",
            failed_test=task.test_code,
        ),
        enriched_labels={"rule_id": rule.rule_id},
        confidence=0.75,
        diagnosis_source="rule_confirmed_by_model",
        prompt_version="confirm-v1",
    )
