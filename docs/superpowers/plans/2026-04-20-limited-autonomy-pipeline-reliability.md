# Limited Autonomy Pipeline Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current diagnose/aggregate/synthesize pipeline reliably produce weakness-aligned specs on the real FaultLens demo, with stable weakness semantics and `synthesize_status=success`.

**Architecture:** Keep the existing constrained three-stage pipeline, but harden it in three places: canonical weakness taxonomy, weakness-specific synthesis policy, and task-level validators plus repair loops. Do not convert the system into a general autonomous agent; only strengthen bounded autonomy inside each stage.

**Tech Stack:** Python 3.11, Pydantic, pytest, Typer CLI, JSONL artifacts, prompt-driven LLM calls

---

## File Structure

### Diagnosis / Taxonomy

**Files:**
- Modify: `codemint/models/diagnosis.py`
- Modify: `codemint/diagnose/pipeline.py`
- Modify: `prompts/diagnose_deep_analysis.txt`
- Modify: `prompts/diagnose_rule_confirm.txt`
- Test: `tests/diagnose/test_pipeline.py`

**Responsibilities:**
- `codemint/models/diagnosis.py`: canonical diagnosis semantics, failure vs non-failure boundary
- `codemint/diagnose/pipeline.py`: real-model normalization and deterministic taxonomy mapping
- prompts: force the model to stay inside the canonical ontology
- tests: lock down taxonomy and non-failure handling

### Aggregation / Canonical Weakness Ontology

**Files:**
- Modify: `codemint/aggregate/pipeline.py`
- Modify: `codemint/aggregate/collective.py`
- Modify: `prompts/aggregate_collective_diagnosis.txt`
- Test: `tests/aggregate/test_pipeline.py`
- Test: `tests/aggregate/test_collective.py`

**Responsibilities:**
- define which tags may merge and which must remain distinct
- preserve only real trainable weaknesses
- prevent non-failure drift from entering weakness reports

### Synthesis Policy / Validators / Repair

**Files:**
- Modify: `codemint/synthesize/generate.py`
- Modify: `codemint/synthesize/feasibility.py`
- Modify: `codemint/synthesize/pipeline.py`
- Modify: `prompts/synthesize_spec_generation.txt`
- Modify: `prompts/synthesize_feasibility_check.txt`
- Test: `tests/synthesize/test_generate.py`
- Test: `tests/synthesize/test_pipeline.py`

**Responsibilities:**
- add weakness-specific generation constraints
- add local validators that check weakness alignment, not just schema
- make retries strategy-aware instead of only reason-aware

### Metadata / Observability

**Files:**
- Modify: `codemint/models/run_metadata.py`
- Modify: `codemint/run/pipeline.py`
- Modify: `codemint/logging.py`
- Test: `tests/test_run_metadata.py`
- Test: `tests/run/test_pipeline.py`

**Responsibilities:**
- expose degradation causes clearly
- summarize uncovered weaknesses and failure reasons
- keep CLI output honest about run quality

### End-to-End Validation

**Files:**
- Test: `tests/integration/test_full_pipeline.py`
- Use runtime artifacts under: `demo_runs/faultlens/`

**Responsibilities:**
- prove the whole pipeline works on fixtures
- prove the real FaultLens demo reaches the expected result

## Task 1: Freeze Canonical Diagnosis Taxonomy

**Files:**
- Modify: `codemint/models/diagnosis.py`
- Modify: `codemint/diagnose/pipeline.py`
- Modify: `prompts/diagnose_deep_analysis.txt`
- Modify: `prompts/diagnose_rule_confirm.txt`
- Test: `tests/diagnose/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- `function_name_mismatch` staying stable as a canonical weakness tag
- `markdown_formatting` staying stable as a canonical weakness tag
- `missing_code_block` not drifting into success-like tags
- `correct_output`, `correct_execution`, `correct_solution`, `pass` normalizing to `non_failure`

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest -q --import-mode=importlib tests/diagnose/test_pipeline.py -q
```

Expected: failures showing current taxonomy drift or missing normalization

- [ ] **Step 3: Tighten diagnose prompt contracts**

Update:
- `prompts/diagnose_deep_analysis.txt`
- `prompts/diagnose_rule_confirm.txt`

Add explicit rules:
- `function_name_mismatch` means public entry-point/test-contract mismatch
- `markdown_formatting` means raw-output formatting violation
- `missing_code_block` means explanation or commentary instead of executable code
- success-like records must use `diagnosis_source=non_failure`

- [ ] **Step 4: Implement deterministic normalization**

In `codemint/diagnose/pipeline.py`:
- normalize synonyms into canonical tags
- normalize success-like outputs into `non_failure`
- preserve stable fault type boundaries

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
pytest -q --import-mode=importlib tests/diagnose/test_pipeline.py -q
```

- [ ] **Step 6: Commit**

```bash
git add codemint/models/diagnosis.py codemint/diagnose/pipeline.py prompts/diagnose_deep_analysis.txt prompts/diagnose_rule_confirm.txt tests/diagnose/test_pipeline.py
git commit -m "fix: stabilize diagnosis taxonomy and non-failure mapping"
```

## Task 2: Freeze Canonical Weakness Merge Policy

**Files:**
- Modify: `codemint/aggregate/pipeline.py`
- Modify: `codemint/aggregate/collective.py`
- Modify: `prompts/aggregate_collective_diagnosis.txt`
- Test: `tests/aggregate/test_pipeline.py`
- Test: `tests/aggregate/test_collective.py`

- [ ] **Step 1: Write the failing tests**

Add tests that lock the intended canonical behavior for:
- `missing_code_block` vs `syntax_error`
- `function_name_mismatch`
- `markdown_formatting`
- non-failure diagnosis exclusion

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest -q --import-mode=importlib tests/aggregate/test_pipeline.py tests/aggregate/test_collective.py -q
```

- [ ] **Step 3: Encode explicit canonical merge policy**

Recommendation:
- keep `function_name_mismatch` distinct
- keep `markdown_formatting` distinct
- decide explicitly whether `missing_code_block` stays distinct or canonicalizes into a broader `non_executable_code/syntax_error` bucket

Implement this policy in:
- `codemint/aggregate/pipeline.py`
- `codemint/aggregate/collective.py`

- [ ] **Step 4: Update aggregate prompt guidance**

In `prompts/aggregate_collective_diagnosis.txt`, specify:
- which tags may merge
- which tags must remain distinct
- that non-failure samples must never become weaknesses

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
pytest -q --import-mode=importlib tests/aggregate/test_pipeline.py tests/aggregate/test_collective.py -q
```

- [ ] **Step 6: Commit**

```bash
git add codemint/aggregate/pipeline.py codemint/aggregate/collective.py prompts/aggregate_collective_diagnosis.txt tests/aggregate/test_pipeline.py tests/aggregate/test_collective.py
git commit -m "fix: stabilize aggregate weakness merge policy"
```

## Task 3: Add Weakness-Specific Synthesis Policy for `function_name_mismatch`

**Files:**
- Modify: `codemint/synthesize/generate.py`
- Modify: `codemint/synthesize/feasibility.py`
- Modify: `prompts/synthesize_spec_generation.txt`
- Test: `tests/synthesize/test_generate.py`
- Test: `tests/synthesize/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add tests requiring:
- `must_cover` explicitly enforces the exact callable entry-point name
- `must_avoid` explicitly forbids alternate public function names
- local feasibility rejects specs without a stable harness contract

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest -q --import-mode=importlib tests/synthesize/test_generate.py tests/synthesize/test_pipeline.py -q
```

- [ ] **Step 3: Add generation augmentation**

In `codemint/synthesize/generate.py`:
- inject exact callable contract requirements for `function_name_mismatch`
- inject forbidden alternate naming patterns into `must_avoid`

- [ ] **Step 4: Add local feasibility validator**

In `codemint/synthesize/feasibility.py`:
- reject any `function_name_mismatch` spec that does not enforce a single exact public entry-point contract

- [ ] **Step 5: Update prompt rules**

In `prompts/synthesize_spec_generation.txt`:
- add weakness-specific guidance for interface/harness mismatches

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
pytest -q --import-mode=importlib tests/synthesize/test_generate.py tests/synthesize/test_pipeline.py -q
```

- [ ] **Step 7: Commit**

```bash
git add codemint/synthesize/generate.py codemint/synthesize/feasibility.py prompts/synthesize_spec_generation.txt tests/synthesize/test_generate.py tests/synthesize/test_pipeline.py
git commit -m "feat: add function-name-mismatch synthesis policy"
```

## Task 4: Add Weakness-Specific Synthesis Policy for `markdown_formatting`

**Files:**
- Modify: `codemint/synthesize/generate.py`
- Modify: `codemint/synthesize/feasibility.py`
- Modify: `prompts/synthesize_spec_generation.txt`
- Test: `tests/synthesize/test_generate.py`
- Test: `tests/synthesize/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add tests requiring:
- key-trap grounding can rely on markdown-fence/raw-code evidence
- `must_cover` explicitly requires raw executable output
- `must_avoid` explicitly forbids markdown fences / wrapping delimiters
- local feasibility rejects formatting specs that do not encode those rules

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest -q --import-mode=importlib tests/synthesize/test_generate.py tests/synthesize/test_pipeline.py -q
```

- [ ] **Step 3: Add generation augmentation**

In `codemint/synthesize/generate.py`:
- add raw-output requirements for `markdown_formatting`
- add fence/delimiter prohibitions to `must_avoid`

- [ ] **Step 4: Expand grounding helpers**

Teach evidence-grounding to recognize:
- backticks
- markdown fences
- raw code / fenced code distinctions

- [ ] **Step 5: Add local feasibility validator**

In `codemint/synthesize/feasibility.py`:
- reject `markdown_formatting` specs that do not require raw executable output

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
pytest -q --import-mode=importlib tests/synthesize/test_generate.py tests/synthesize/test_pipeline.py -q
```

- [ ] **Step 7: Commit**

```bash
git add codemint/synthesize/generate.py codemint/synthesize/feasibility.py prompts/synthesize_spec_generation.txt tests/synthesize/test_generate.py tests/synthesize/test_pipeline.py
git commit -m "feat: add markdown-formatting synthesis policy"
```

## Task 5: Make Regeneration Strategy-Aware

**Files:**
- Modify: `codemint/synthesize/pipeline.py`
- Modify: `codemint/synthesize/generate.py`
- Test: `tests/synthesize/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add tests verifying that retries carry:
- structured repair mode
- feasibility failure reason
- diversity failure reason when relevant

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest -q --import-mode=importlib tests/synthesize/test_pipeline.py -q
```

- [ ] **Step 3: Implement repair-mode mapping**

In `codemint/synthesize/pipeline.py`:
- map validator failures into stable repair modes like:
  - `contract_mismatch`
  - `raw_output_required`
  - `executable_code_required`
  - `duplicate_diversity_pattern`

- [ ] **Step 4: Include repair context in generation payload**

In `codemint/synthesize/generate.py`:
- add repair context to payload
- keep payload backward-compatible for default stub behavior

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
pytest -q --import-mode=importlib tests/synthesize/test_pipeline.py -q
```

- [ ] **Step 6: Commit**

```bash
git add codemint/synthesize/pipeline.py codemint/synthesize/generate.py tests/synthesize/test_pipeline.py
git commit -m "feat: make synthesis regeneration strategy-aware"
```

## Task 6: Add Task-Level Weakness Validators

**Files:**
- Modify: `codemint/synthesize/feasibility.py`
- Test: `tests/synthesize/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add validators for:
- `syntax_error`: spec must enforce syntactically complete executable code
- `function_name_mismatch`: spec must verify exact callable entry point
- `markdown_formatting`: spec must require raw executable output
- canonicalized `missing_code_block/non_executable_code`: spec must test for executable output presence

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest -q --import-mode=importlib tests/synthesize/test_pipeline.py -q
```

- [ ] **Step 3: Implement validators**

In `codemint/synthesize/feasibility.py`:
- add weakness-specific checks beyond schema
- keep them local and deterministic

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest -q --import-mode=importlib tests/synthesize/test_pipeline.py -q
```

- [ ] **Step 5: Commit**

```bash
git add codemint/synthesize/feasibility.py tests/synthesize/test_pipeline.py
git commit -m "feat: add weakness-specific synthesis validators"
```

## Task 7: Improve Metadata for Actionable Degradation

**Files:**
- Modify: `codemint/models/run_metadata.py`
- Modify: `codemint/run/pipeline.py`
- Modify: `codemint/logging.py`
- Test: `tests/test_run_metadata.py`
- Test: `tests/run/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add tests for metadata fields:
- `attempted_weaknesses`
- `covered_weaknesses`
- `synthesize_failure_reasons_by_weakness`

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest -q --import-mode=importlib tests/test_run_metadata.py tests/run/test_pipeline.py -q
```

- [ ] **Step 3: Implement summary fields**

In `codemint/run/pipeline.py`:
- summarize errors per weakness
- record attempted vs covered weakness sets

- [ ] **Step 4: Update CLI summary**

In `codemint/logging.py`:
- show degraded cause compactly without overwhelming the line

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
pytest -q --import-mode=importlib tests/test_run_metadata.py tests/run/test_pipeline.py -q
```

- [ ] **Step 6: Commit**

```bash
git add codemint/models/run_metadata.py codemint/run/pipeline.py codemint/logging.py tests/test_run_metadata.py tests/run/test_pipeline.py
git commit -m "feat: improve degraded synthesis observability"
```

## Task 8: Revalidate Full Test Suite

**Files:**
- Test: full suite

- [ ] **Step 1: Run the full suite**

Run:
```bash
pytest -q --import-mode=importlib
```

Expected: all green

- [ ] **Step 2: Fix any regression before proceeding**

Do not proceed to real demo validation with red tests

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "test: stabilize pipeline after weakness-specific synthesis improvements"
```

## Task 9: Revalidate Real FaultLens Demo

**Files:**
- Use: `demo_runs/faultlens/demo.yaml`
- Use: `demo_runs/faultlens/codemint_inference.jsonl`
- Use: `demo_runs/faultlens/codemint_results.jsonl`

- [ ] **Step 1: Clean run directory**

Run:
```bash
rm -rf demo_runs/faultlens/artifacts/full-real
```

- [ ] **Step 2: Run full real pipeline**

Run:
```bash
python3.11 -m codemint.cli run \
  demo_runs/faultlens/codemint_inference.jsonl \
  demo_runs/faultlens/codemint_results.jsonl \
  --output-root demo_runs/faultlens/artifacts \
  --run-id full-real \
  --config demo_runs/faultlens/demo.yaml
```

- [ ] **Step 3: Inspect artifacts**

Check:
- `demo_runs/faultlens/artifacts/full-real/run_metadata.json`
- `demo_runs/faultlens/artifacts/full-real/weaknesses.json`
- `demo_runs/faultlens/artifacts/full-real/specs.jsonl`
- `demo_runs/faultlens/artifacts/full-real/errors.jsonl`

- [ ] **Step 4: Apply acceptance checklist**

Acceptance requires:
- `synthesize_status == "success"`
- `weaknesses_without_specs == []`
- no pseudo-weakness like `correct_output` or `correct_execution`
- at least one valid spec for each real top weakness
- no repeated single failure mode dominating one weakness without repair escalation

- [ ] **Step 5: Commit the validated improvements**

```bash
git add .
git commit -m "feat: bring limited-autonomy pipeline to reliable real-demo output"
```

## Acceptance Criteria

The system is only considered aligned with expectations when all are true:

- Real demo ends with `synthesize_status=success`
- `weaknesses_without_specs=[]`
- No pseudo-weakness from successful examples enters the report
- `syntax_error` has at least one valid spec
- `function_name_mismatch` has at least one valid spec
- `markdown_formatting` has at least one valid spec
- Full automated suite passes

