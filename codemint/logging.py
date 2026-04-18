from __future__ import annotations

from codemint.models.run_metadata import RunMetadata
from codemint.run.dry_run import DryRunEstimate


def format_dry_run_summary(estimate: DryRunEstimate) -> str:
    lines = [estimate.summary_line]
    for stage in ("diagnose", "aggregate", "synthesize"):
        calls = estimate.stage_calls[stage]
        label = "call" if calls == 1 else "calls"
        lines.append(
            f"{stage}: {calls} {label}, ~{estimate.stage_tokens[stage]} tokens, "
            f"~{estimate.stage_seconds[stage]}s"
        )
    return "\n".join(lines)


def format_run_summary(metadata: RunMetadata) -> str:
    return (
        f"Run {metadata.run_id}: stages={','.join(metadata.stages_executed) or 'none'}, "
        f"diagnosed={metadata.summary.diagnosed}, "
        f"weaknesses={metadata.summary.weaknesses_found}, "
        f"specs={metadata.summary.specs_generated}"
    )
