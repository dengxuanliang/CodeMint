from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TaskRecord:
    task_id: int
    content: str
    canonical_solution: str
    completion: str
    test_code: str
    labels: dict[str, Any]
    accepted: bool
    metrics: dict[str, Any]
    extra: dict[str, Any]
