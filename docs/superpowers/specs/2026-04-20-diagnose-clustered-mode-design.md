# Diagnose Clustered Mode Design

**Problem:** `diagnose` currently scales linearly with the number of evaluation samples because the primary path is item-by-item diagnosis. This preserves quality for small and medium runs, but it becomes unnecessarily expensive for large logs that contain many repeated or near-duplicate failures.

**Goal:** Add a `clustered` diagnose mode that reduces repeated model work on large logs while preserving the current item-level output contract and keeping the existing `item` mode as the default behavior.

**Non-Goals:**
- Do not change the default diagnose behavior.
- Do not add an `auto` mode.
- Do not move single-sample semantic repair into `aggregate`.
- Do not change the output schema consumed by `aggregate` or `synthesize`.
- Do not add a full evaluation harness or advanced observability system in this iteration.

## Requirements

- `diagnose.processing_mode` supports exactly `item | clustered`, with default `item`.
- `item` mode must remain behaviorally compatible with the current implementation.
- `clustered` mode must still emit item-level `DiagnosisRecord` rows for every task.
- `aggregate` and `synthesize` must remain mode-agnostic.
- `clustered` mode must support selective fallback to item-level diagnosis when cluster propagation is unsafe.
- The system must emit minimal run statistics for clustered operation.

## Recommended Architecture

Keep the public `run_diagnose()` contract unchanged and introduce a strategy split inside the diagnose package:

- `pipeline.py`
  Routes to either `item` or `clustered` mode and owns shared output persistence.
- `item_mode.py`
  Hosts the current single-item diagnosis flow so the default path is isolated and regression-protected.
- `fingerprint.py`
  Normalizes task evidence and produces stable cluster fingerprints.
- `clustering.py`
  Builds exact and near-duplicate clusters and selects representative samples.
- `propagation.py`
  Propagates representative diagnoses to cluster members and identifies members that require item fallback.
- `clustered_mode.py`
  Orchestrates clustered diagnosis, representative item diagnosis, propagation, fallback, and optional cluster artifacts.

This structure keeps the new complexity inside `codemint/diagnose/` and prevents downstream modules from being coupled to execution mode.

## Data Flow

### Item Mode

Unchanged current flow:

1. Load task
2. Rule screening
3. Optional rule confirmation / deep analysis
4. Normalize taxonomy and source fields
5. Write item-level `DiagnosisRecord`

### Clustered Mode

1. Normalize task text and extract stable signals
2. Generate fingerprints for each task
3. Build exact and near-duplicate clusters
4. Select representative samples per cluster
5. Run the existing item diagnosis flow on representatives
6. Propagate diagnoses to safe cluster members
7. Fallback to item diagnosis for unsafe members
8. Emit full item-level `DiagnosisRecord` rows
9. Optionally persist cluster debug artifacts

The core rule is that clustered mode may compress *work*, but it may not compress the final diagnosis artifact contract.

## Clustering Strategy

The first iteration should stay conservative:

- Exact clustering first, based on normalized fingerprints.
- Near-duplicate clustering second, gated by a configurable similarity threshold.
- Representative count capped by config.
- Propagation only allowed when representative diagnoses are sufficiently consistent.
- A per-cluster propagation size cap prevents one weak cluster from labeling an overly large population.

The fingerprint should favor robustness over semantic ambition. It should be constructed from:

- rule-match hints
- output-format hints
- entry-point hints
- failed-test/assertion templates
- normalized completion snippets

This is enough to collapse the most repetitive failure modes without introducing broad semantic guesswork.

## Fallback Rules

Selective fallback should trigger when propagation is not trustworthy. The first iteration should implement only a small, explainable rule set:

- representative primary tags disagree
- high-priority rule evidence conflicts inside the same cluster
- cluster similarity falls below the configured threshold
- propagated confidence is below the configured low-confidence threshold

Fallback members return to the existing item diagnosis flow. This is the main safety valve that preserves quality.

## Configuration

Add the following fields under diagnose configuration:

```yaml
diagnose:
  processing_mode: item
  cluster_representatives: 3
  clustering_threshold: 0.85
  low_confidence_threshold: 0.55
  rediagnose_low_confidence: true
  max_cluster_size_for_propagation: 50
```

Defaults must preserve current behavior by keeping `processing_mode: item`.

## Artifacts and Metadata

Keep existing artifacts unchanged:

- `diagnoses.jsonl`
- `weaknesses.json`
- `specs.jsonl`

Resumption semantics stay unchanged:

- `diagnoses.jsonl` remains the only per-task checkpoint source and resumes by missing `task_id`.
- `weaknesses.json` and `specs.jsonl` remain downstream derived artifacts that can be rebuilt from complete diagnose output.

Add one optional debug artifact for clustered mode:

- `diagnose_clusters.json`

This file should contain cluster ids, members, representatives, fingerprint summaries, and fallback members. It is for debugging only and must not become a downstream dependency.

Add minimal run metadata:

- `diagnose_processing_mode`
- `cluster_count`
- `compression_ratio`
- `representative_diagnoses`
- `propagated_diagnoses`
- `fallback_item_diagnoses`

These fields are enough to explain clustered runs without turning this iteration into a full observability project.

## Testing Strategy

Testing should be layered:

- regression tests proving `item` mode behavior is unchanged
- unit tests for fingerprint generation
- unit tests for cluster construction
- unit tests for propagation and fallback
- integration tests for clustered diagnose output shape
- run pipeline tests proving metadata and artifacts are emitted correctly

The most important regression guarantee is that default users who do not opt into clustered mode see no behavior change.

## Risks and Mitigations

- **Risk:** Cluster propagation introduces incorrect diagnoses.
  - **Mitigation:** conservative fallback rules, low-confidence re-diagnosis, max propagation cluster size.

- **Risk:** Downstream stages accidentally start depending on clustered internals.
  - **Mitigation:** keep final output contract identical and debug artifact explicitly optional.

- **Risk:** Default behavior changes unintentionally.
  - **Mitigation:** isolate `item_mode`, keep `processing_mode=item` by default, and add regression coverage.

## Decision Summary

- Keep `item` as the default and unchanged path.
- Add `clustered` as an internal diagnose strategy only.
- Preserve item-level `DiagnosisRecord` output in both modes.
- Limit this iteration to clustered diagnosis, minimal metadata, and no broader evaluation framework.
