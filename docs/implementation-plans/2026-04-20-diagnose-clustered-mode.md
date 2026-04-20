# Diagnose Clustered Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `clustered` diagnose mode for large evaluation logs while preserving `item` as the default mode and keeping downstream contracts unchanged.

**Architecture:** Split the current diagnose logic into a stable `item` strategy and a new `clustered` strategy inside `codemint/diagnose/`. Clustered mode will compress repeated work through fingerprinting, clustering, representative diagnosis, propagation, and selective fallback, while still emitting item-level `DiagnosisRecord` rows.

**Tech Stack:** Python 3.12, `pytest`, `pydantic`, existing CodeMint config/models/pipeline modules.

---

## Proposed File Structure

- `codemint/config.py`: add diagnose-mode and clustered-mode configuration.
- `codemint/diagnose/pipeline.py`: route by `processing_mode` and keep shared persistence.
- `codemint/diagnose/item_mode.py`: extracted current item-by-item diagnosis flow.
- `codemint/diagnose/fingerprint.py`: task normalization and fingerprint extraction.
- `codemint/diagnose/clustering.py`: cluster models, exact clustering, near-duplicate grouping, representative selection.
- `codemint/diagnose/propagation.py`: propagation and fallback decisions.
- `codemint/diagnose/clustered_mode.py`: clustered orchestration and optional cluster artifact output.
- `codemint/models/run_metadata.py`: add minimal diagnose clustered statistics.
- `codemint/run/pipeline.py`: persist diagnose-mode summary fields.
- `tests/diagnose/test_item_mode.py`: item-mode regression coverage.
- `tests/diagnose/test_fingerprint.py`: fingerprint tests.
- `tests/diagnose/test_clustering.py`: clustering tests.
- `tests/diagnose/test_clustered_mode.py`: clustered-mode tests.
- `tests/run/test_pipeline.py`: run metadata and artifact integration coverage.

### Task 1: Extract Stable Item Diagnose Flow

**Files:**
- Create: `codemint/diagnose/item_mode.py`
- Modify: `codemint/diagnose/pipeline.py`
- Test: `tests/diagnose/test_item_mode.py`
- Test: `tests/diagnose/test_pipeline.py`

- [ ] **Step 1: Write the failing regression tests for item-mode extraction**

```python
def test_item_mode_matches_existing_pipeline_behavior(tmp_path):
    tasks = [...]
    output = tmp_path / "diagnoses.jsonl"

    pipeline_result = run_diagnose(tasks, output)
    item_result = run_item_mode(tasks, output_path=tmp_path / "item.jsonl")

    assert [row.model_dump(mode="json") for row in pipeline_result] == [
        row.model_dump(mode="json") for row in item_result
    ]
```

- [ ] **Step 2: Run the focused tests to verify failure**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_item_mode.py tests/diagnose/test_pipeline.py`
Expected: FAIL because `run_item_mode` does not exist yet.

- [ ] **Step 3: Implement minimal item-mode extraction**

```python
def run_item_mode(...):
    # Move current per-task diagnosis flow here without changing behavior.
```

- [ ] **Step 4: Route pipeline through item mode by default**

```python
if resolved_config.diagnose.processing_mode == "item":
    return run_item_mode(...)
```

- [ ] **Step 5: Run tests to verify item behavior is preserved**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_item_mode.py tests/diagnose/test_pipeline.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add codemint/diagnose/item_mode.py codemint/diagnose/pipeline.py tests/diagnose/test_item_mode.py tests/diagnose/test_pipeline.py
git commit -m "refactor: extract item diagnose mode"
```

### Task 2: Add Diagnose Mode Configuration

**Files:**
- Modify: `codemint/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests for clustered diagnose settings**

```python
def test_config_supports_clustered_diagnose_mode():
    config = load_config_from_string(
        '''
        diagnose:
          processing_mode: clustered
          cluster_representatives: 2
        '''
    )
    assert config.diagnose.processing_mode == "clustered"
    assert config.diagnose.cluster_representatives == 2
```

- [ ] **Step 2: Run the config test to verify failure**

Run: `pytest --import-mode=importlib -q tests/test_config.py`
Expected: FAIL because diagnose clustered settings are missing.

- [ ] **Step 3: Implement diagnose clustered config fields with safe defaults**

```python
class DiagnoseConfig(BaseModel):
    processing_mode: Literal["item", "clustered"] = "item"
    cluster_representatives: int = 3
    clustering_threshold: float = 0.85
    low_confidence_threshold: float = 0.55
    rediagnose_low_confidence: bool = True
    max_cluster_size_for_propagation: int = 50
```

- [ ] **Step 4: Run config tests**

Run: `pytest --import-mode=importlib -q tests/test_config.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codemint/config.py tests/test_config.py
git commit -m "feat: add diagnose clustered configuration"
```

### Task 3: Implement Fingerprint Extraction

**Files:**
- Create: `codemint/diagnose/fingerprint.py`
- Test: `tests/diagnose/test_fingerprint.py`

- [ ] **Step 1: Write failing tests for fingerprint normalization**

```python
def test_fingerprint_captures_entry_point_and_format_hints():
    task = TaskRecord(...)
    fp = build_fingerprint(task)
    assert fp.entry_point_hint == "solve"
    assert fp.output_format_hint == "markdown_fence"
```

- [ ] **Step 2: Run the fingerprint tests to verify failure**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_fingerprint.py`
Expected: FAIL because fingerprint module does not exist.

- [ ] **Step 3: Implement normalized fingerprint extraction**

```python
@dataclass(frozen=True)
class DiagnoseFingerprint:
    ...
```

- [ ] **Step 4: Run fingerprint tests**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_fingerprint.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codemint/diagnose/fingerprint.py tests/diagnose/test_fingerprint.py
git commit -m "feat: add diagnose fingerprints"
```

### Task 4: Implement Clustering and Representative Selection

**Files:**
- Create: `codemint/diagnose/clustering.py`
- Test: `tests/diagnose/test_clustering.py`

- [ ] **Step 1: Write failing tests for exact and near clustering**

```python
def test_cluster_tasks_groups_similar_fingerprints():
    tasks = [...]
    fingerprints = [...]
    clusters = cluster_tasks(tasks, fingerprints, threshold=0.85)
    assert len(clusters) == 2
    assert clusters[0].representative_task_ids
```

- [ ] **Step 2: Run clustering tests to verify failure**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_clustering.py`
Expected: FAIL because clustering module does not exist.

- [ ] **Step 3: Implement cluster data structures and grouping**

```python
@dataclass
class DiagnoseCluster:
    cluster_id: str
    member_task_ids: list[int]
    representative_task_ids: list[int]
```

- [ ] **Step 4: Run clustering tests**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_clustering.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codemint/diagnose/clustering.py tests/diagnose/test_clustering.py
git commit -m "feat: add diagnose clustering"
```

### Task 5: Implement Propagation and Selective Fallback

**Files:**
- Create: `codemint/diagnose/propagation.py`
- Test: `tests/diagnose/test_clustered_mode.py`

- [ ] **Step 1: Write failing tests for safe propagation and fallback**

```python
def test_propagation_falls_back_when_representatives_disagree():
    representatives = [...]
    result = propagate_cluster_diagnoses(...)
    assert result.fallback_task_ids == [2, 3]
```

- [ ] **Step 2: Run clustered-mode tests to verify failure**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_clustered_mode.py`
Expected: FAIL because propagation logic does not exist.

- [ ] **Step 3: Implement propagation result and fallback rules**

```python
@dataclass
class PropagationResult:
    propagated: list[DiagnosisRecord]
    fallback_task_ids: list[int]
```

- [ ] **Step 4: Run clustered-mode tests**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_clustered_mode.py`
Expected: PASS for propagation coverage

- [ ] **Step 5: Commit**

```bash
git add codemint/diagnose/propagation.py tests/diagnose/test_clustered_mode.py
git commit -m "feat: add diagnose propagation fallback rules"
```

### Task 6: Assemble Clustered Diagnose Mode

**Files:**
- Create: `codemint/diagnose/clustered_mode.py`
- Modify: `codemint/diagnose/pipeline.py`
- Test: `tests/diagnose/test_clustered_mode.py`
- Test: `tests/diagnose/test_pipeline.py`

- [ ] **Step 1: Write failing integration tests for clustered mode**

```python
def test_run_diagnose_clustered_emits_item_level_records(tmp_path):
    config = CodeMintConfig.model_validate({"diagnose": {"processing_mode": "clustered"}})
    results = run_diagnose(tasks, tmp_path / "diagnoses.jsonl", config=config)
    assert len(results) == len(tasks)
    assert all(result.task_id for result in results)
```

- [ ] **Step 2: Run clustered diagnose tests to verify failure**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_clustered_mode.py tests/diagnose/test_pipeline.py`
Expected: FAIL because clustered orchestration does not exist.

- [ ] **Step 3: Implement clustered orchestration**

```python
def run_clustered_mode(...):
    fingerprints = ...
    clusters = ...
    representative_diagnoses = ...
    propagated = ...
    fallback = ...
    return ordered_item_level_results
```

- [ ] **Step 4: Persist optional cluster debug artifact**

```python
clusters_path = output_path.parent / "diagnose_clusters.json"
```

- [ ] **Step 5: Run clustered diagnose tests**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_clustered_mode.py tests/diagnose/test_pipeline.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add codemint/diagnose/clustered_mode.py codemint/diagnose/pipeline.py tests/diagnose/test_clustered_mode.py tests/diagnose/test_pipeline.py
git commit -m "feat: add clustered diagnose mode"
```

### Task 7: Add Minimal Run Metadata for Clustered Runs

**Files:**
- Modify: `codemint/models/run_metadata.py`
- Modify: `codemint/run/pipeline.py`
- Test: `tests/run/test_pipeline.py`
- Test: `tests/test_run_metadata.py`

- [ ] **Step 1: Write failing tests for clustered diagnose summary fields**

```python
def test_run_metadata_includes_clustered_diagnose_stats(tmp_path):
    ...
    assert metadata["summary"]["diagnose_processing_mode"] == "clustered"
    assert metadata["summary"]["cluster_count"] == 2
```

- [ ] **Step 2: Run run-pipeline tests to verify failure**

Run: `pytest --import-mode=importlib -q tests/run/test_pipeline.py tests/test_run_metadata.py`
Expected: FAIL because clustered summary fields are missing.

- [ ] **Step 3: Implement minimal clustered metadata plumbing**

```python
class RunSummary(...):
    diagnose_processing_mode: str = "item"
    cluster_count: int = 0
    compression_ratio: float = 1.0
    representative_diagnoses: int = 0
    propagated_diagnoses: int = 0
    fallback_item_diagnoses: int = 0
```

- [ ] **Step 4: Run run metadata tests**

Run: `pytest --import-mode=importlib -q tests/run/test_pipeline.py tests/test_run_metadata.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codemint/models/run_metadata.py codemint/run/pipeline.py tests/run/test_pipeline.py tests/test_run_metadata.py
git commit -m "feat: add clustered diagnose run metadata"
```

### Task 8: Full Regression Slice

**Files:**
- Modify: `tests/diagnose/test_pipeline.py`
- Modify: `tests/run/test_pipeline.py`
- Modify: `tests/test_config.py`
- Modify: `tests/integration/test_full_pipeline.py`

- [ ] **Step 1: Add regression tests for default user no-change behavior**

```python
def test_default_processing_mode_remains_item():
    config = CodeMintConfig()
    assert config.diagnose.processing_mode == "item"
```

- [ ] **Step 2: Run the focused regression suite**

Run: `pytest --import-mode=importlib -q tests/diagnose tests/run/test_pipeline.py tests/test_config.py tests/integration/test_full_pipeline.py`
Expected: PASS with clustered and item coverage.

- [ ] **Step 3: Run the broader targeted suite**

Run: `pytest --import-mode=importlib -q tests/diagnose/test_pipeline.py tests/diagnose/test_item_mode.py tests/diagnose/test_fingerprint.py tests/diagnose/test_clustering.py tests/diagnose/test_clustered_mode.py tests/run/test_pipeline.py tests/test_run_metadata.py tests/test_config.py tests/integration/test_full_pipeline.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/diagnose/test_pipeline.py tests/run/test_pipeline.py tests/test_config.py tests/integration/test_full_pipeline.py
git commit -m "test: cover clustered diagnose end-to-end"
```
