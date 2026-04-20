from __future__ import annotations

from codemint.models.run_metadata import RunMetadata, RunProgressEvent
from codemint.run.dry_run import DryRunEstimate


def format_dry_run_summary(estimate: DryRunEstimate) -> str:
    lines = [estimate.summary_line]
    lines.append(
        f"Estimated tokens: ~{estimate.estimated_input_tokens} input, ~{estimate.estimated_output_tokens} output"
    )
    lines.append(f"Rule-screened (no model call): {estimate.rule_screened}/{estimate.input_count}")
    lines.append(f"Estimated time: ~{estimate.estimated_seconds}s (concurrency={estimate.concurrency})")
    for stage in ("diagnose", "aggregate", "synthesize"):
        calls = estimate.stage_calls[stage]
        label = "call" if calls == 1 else "calls"
        lines.append(
            f"{stage}: {calls} {label}, ~{estimate.stage_input_tokens[stage]} input tokens, "
            f"~{estimate.stage_output_tokens[stage]} output tokens, "
            f"~{estimate.stage_seconds[stage]}s"
        )
    return "\n".join(lines)


def format_run_summary(metadata: RunMetadata) -> str:
    return (
        f"Run {metadata.run_id}: stages={','.join(metadata.stages_executed) or 'none'}, "
        f"diagnosed={metadata.summary.diagnosed}, non_failures={metadata.summary.non_failures}, skipped={metadata.summary.skipped}, "
        f"errors={metadata.summary.errors}, elapsed={metadata.summary.elapsed_seconds:.2f}s, "
        f"weaknesses={metadata.summary.weaknesses_found}, "
        f"specs={metadata.summary.specs_generated}, "
        f"synth_failures={metadata.summary.synthesize_failures}, "
        f"synth_status={metadata.summary.synthesize_status}, "
        f"missing={','.join(metadata.summary.weaknesses_without_specs) or 'none'}"
    )


def format_progress_event(event: RunProgressEvent) -> str:
    total = max(event.total, 1)
    percent = int((event.processed / total) * 100) if event.total > 0 else 100
    bar_width = 10
    filled = min((percent * bar_width) // 100, bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)
    eta = "--" if event.eta_seconds is None else f"{event.eta_seconds}s"
    return (
        f"[{event.stage}] {bar} {event.processed}/{event.total} ({percent}%) | "
        f"{event.errors} errors | ETA {eta}"
    )
