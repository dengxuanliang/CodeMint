from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskRecord:
    task_id: int
    content: str
    canonical_solution: str
    completion: str
    test_code: str
    labels: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
