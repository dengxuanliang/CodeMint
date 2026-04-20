from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from codemint.models.task import TaskRecord


class BaseLoader(ABC):
    @abstractmethod
    def load(self, paths: list[Path]) -> list[TaskRecord]:
        raise NotImplementedError


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records
