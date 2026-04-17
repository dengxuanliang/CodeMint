from __future__ import annotations

from collections.abc import Callable

from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.task import TaskRecord


DeepAnalyzer = Callable[[TaskRecord], DiagnosisRecord]


def deep_analyze_with_model(task: TaskRecord, analyzer: DeepAnalyzer) -> DiagnosisRecord:
    return analyzer(task)
