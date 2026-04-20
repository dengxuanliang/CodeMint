from __future__ import annotations

from pathlib import Path

from codemint.loaders.base import BaseLoader, read_jsonl
from codemint.models.task import TaskRecord


class MergedFileLoader(BaseLoader):
    def load(self, paths: list[Path]) -> list[TaskRecord]:
        if len(paths) != 1:
            raise ValueError("MergedFileLoader expects exactly one input file")

        return [TaskRecord(**record) for record in read_jsonl(paths[0])]
