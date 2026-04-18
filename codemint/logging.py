from __future__ import annotations

from codemint.models.run_metadata import RunMetadata
from codemint.run.dry_run import DryRunEstimate


def format_dry_run_summary(estimate: DryRunEstimate) -> str:
    return estimate.summary_line


def format_run_summary(metadata: RunMetadata) -> str:
    return (
        f"Run {metadata.run_id}: stages={','.join(metadata.stages_executed) or 'none'}, "
        f"diagnosed={metadata.summary.diagnosed}, "
        f"weaknesses={metadata.summary.weaknesses_found}, "
        f"specs={metadata.summary.specs_generated}"
    )
