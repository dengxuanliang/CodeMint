from __future__ import annotations

from pathlib import Path

from codemint.config import CodeMintConfig
from codemint.diagnose.confirm import ConfirmAnalyzer
from codemint.diagnose.deep import DeepAnalyzer
from codemint.models.diagnosis import DiagnosisRecord
from codemint.models.task import TaskRecord
from codemint.rules.builtin import DiagnosisRule


def run_item_mode(
    tasks: list[TaskRecord],
    output_path: Path,
    rules: list[DiagnosisRule] | None = None,
    *,
    config: CodeMintConfig | None = None,
    confirm_analyzer: ConfirmAnalyzer | None = None,
    deep_analyzer: DeepAnalyzer | None = None,
) -> list[DiagnosisRecord]:
    from codemint.diagnose import pipeline as pipeline_module

    active_rules = rules or pipeline_module.build_rules()
    resolved_config = config or CodeMintConfig()
    confirmer = confirm_analyzer or pipeline_module._default_confirm_analyzer(resolved_config)
    deep = deep_analyzer or pipeline_module._default_deep_analyzer(resolved_config)
    engine = pipeline_module.RuleEngine(active_rules)
    pipeline_module._validate_unique_task_ids(tasks)
    existing_diagnoses = pipeline_module._load_existing_diagnoses(output_path)

    missing_task_ids = set(
        pipeline_module.find_missing_task_ids(output_path, [task.task_id for task in tasks])
    )
    new_diagnoses: list[DiagnosisRecord] = []
    for task in tasks:
        if task.task_id not in missing_task_ids:
            continue
        new_diagnoses.append(pipeline_module._diagnose_task(task, engine, confirmer, deep))

    if new_diagnoses:
        pipeline_module.append_jsonl(
            output_path,
            [diagnosis.model_dump(mode="json") for diagnosis in new_diagnoses],
        )

    return existing_diagnoses + new_diagnoses
