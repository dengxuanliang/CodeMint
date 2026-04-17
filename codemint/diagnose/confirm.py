from __future__ import annotations

from collections.abc import Callable

from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.builtin import DiagnosisRule


ConfirmAnalyzer = Callable[[TaskRecord, str], DiagnosisRecord]


def confirm_rule_with_model(
    task: TaskRecord,
    rule: DiagnosisRule,
    analyzer: ConfirmAnalyzer,
) -> DiagnosisRecord:
    return analyzer(task, rule.rule_id)
