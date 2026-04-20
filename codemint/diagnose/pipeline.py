from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from codemint.config import CodeMintConfig
from codemint.diagnose.confirm import ConfirmAnalyzer
from codemint.diagnose.deep import DeepAnalyzer
from codemint.diagnose.item_mode import run_item_mode
from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.builtin import DiagnosisRule


def run_diagnose(
    tasks: list[TaskRecord],
    output_path: Path,
    rules: list[DiagnosisRule] | None = None,
    *,
    config: CodeMintConfig | None = None,
    confirm_analyzer: ConfirmAnalyzer | None = None,
    deep_analyzer: DeepAnalyzer | None = None,
) -> list[DiagnosisRecord]:
    resolved_config = config or CodeMintConfig()
    processing_mode = getattr(getattr(resolved_config, "diagnose", None), "processing_mode", "item")
    if processing_mode == "item":
        return run_item_mode(
            tasks,
            output_path,
            rules=rules,
            config=resolved_config,
            confirm_analyzer=confirm_analyzer,
            deep_analyzer=deep_analyzer,
        )

    raise ValueError(f"Unsupported diagnose processing mode: {processing_mode}")
