# Synthesize Input Contract Tightening Design

## Scope

This design covers one bounded change only:

- tighten the `synthesize` input contract so it relies on stable structural fields
- keep a very weak, locally generated narrative helper
- stop passing upstream free-form aggregate narrative directly into synthesize

This design does not cover broader synthesize prompt redesign, provider tuning, or changes to diagnose/aggregate taxonomy.

## Problem

The current synthesize path still consumes unstable upstream text:

- `weakness.collective_diagnosis.refined_root_cause`
- `weakness.collective_diagnosis.capability_cliff`
- merged multi-sample evidence in some fallback paths

This creates two product risks:

1. repeated runs on the same input can produce different spec framing even when canonical weakness identity is already stable
2. aggregate narrative drift leaks into synthesize, so synthesize is implicitly depending on a field that is not contract-stable

After aggregate responsibility was tightened, the main remaining instability is no longer cluster identity. It is the downstream consumption of narrative-rich text.

## Goal

Make synthesize depend primarily on stable fields:

- `fault_type`
- primary `sub_tag`
- `frequency`
- representative evidence

Keep one weak helper field:

- `canonical_summary`

`canonical_summary` must be locally generated and deterministic. It may help problem framing, but it must not control feasibility or override weakness-specific rules.

## Non-Goals

- do not remove evidence grounding
- do not redesign the spec schema
- do not remove diversity logic
- do not change top-n allocation policy
- do not make synthesize fully deterministic end-to-end

## Design

### 1. Introduce a Local Synthesis Input View

Add a local synthesize-side view object, for example:

```python
class SynthesisInputView(StrictModel):
    fault_type: FaultType
    primary_sub_tag: str
    frequency: int
    sample_task_ids: list[int]
    canonical_summary: str
    representative_evidence: dict[str, str]
```

This view is derived locally from:

- `WeaknessEntry`
- one representative evidence row for that weakness

This becomes the only weakness payload passed into `generate_spec()`.

### 2. Strong vs Weak Inputs

#### Strong inputs

These fields are allowed to define spec structure:

- `fault_type`
- `primary_sub_tag`
- `frequency`
- `sample_task_ids`
- representative evidence:
  - `wrong_line`
  - `correct_approach`
  - `failed_test`

These fields drive:

- weakness-specific must-cover rules
- must-avoid rules
- key-trap grounding
- feasibility validation
- entrypoint extraction

#### Weak input

Only one weak helper is allowed:

- `canonical_summary`

Rules:

- one sentence only
- locally generated
- deterministic
- used only for high-level problem framing
- never used by feasibility logic
- never used to infer primary weakness identity

### 3. Remove Direct Narrative Pass-Through

Do not pass these fields into synthesize model payloads:

- `collective_diagnosis.refined_root_cause`
- `collective_diagnosis.capability_cliff`
- `misdiagnosed_ids`
- `misdiagnosis_corrections`
- diagnosis-level free-form `description`
- merged long-form cluster evidence text

These fields may still exist in upstream artifacts for observability, but they are outside the synthesize contract.

### 4. Canonical Summary Generation

`canonical_summary` should be generated locally by deterministic rules.

#### Fixed summaries for stable canonical weaknesses

- `missing_code_block`
  - `Model outputs explanation or transformed prompt instead of executable code.`
- `function_name_mismatch`
  - `Model violates the required public entrypoint contract.`
- `markdown_formatting`
  - `Model wraps otherwise executable code in markdown fences.`
- `syntax_error`
  - `Model emits code that is not syntactically executable.`
- `non_executable_code`
  - `Model emits code-like output that still cannot run directly.`

#### Controlled summary for `logic_error`

`logic_error` needs slightly more abstraction, but still must remain deterministic.

Recommended approach:

- derive one coarse bucket from representative evidence
- map that bucket to a fixed sentence

Allowed local buckets:

- output contract mismatch
- edge case failure
- return structure mismatch
- algorithmic semantic error
- API or data contract misuse

Example fixed summaries:

- `Model returns an output that violates the expected contract.`
- `Model fails on a required boundary or edge condition.`
- `Model returns the wrong structure despite executable code.`
- `Model computes the wrong result with executable logic.`
- `Model misuses an API or data contract in executable code.`

Important constraint:

- no LLM call is used to create `canonical_summary`
- no free-form multi-sentence local summary generation

### 5. Payload Changes in `generate_spec`

Current weakness payload effectively includes:

- `fault_type`
- `sub_tags`
- `root_cause`
- `capability_cliff`

Replace that with:

- `fault_type`
- `primary_sub_tag`
- `frequency`
- `sample_task_ids`
- `canonical_summary`

Keep `original_evidence`, but ensure it is one representative evidence row, not merged narrative text.

### 6. Feasibility Contract Boundary

Feasibility logic must continue to depend on:

- target weakness canonical tag
- representative evidence
- generated spec fields

Feasibility logic must not read:

- `canonical_summary`
- aggregate narrative fields

This preserves a clean separation:

- `canonical_summary` helps phrasing
- evidence and canonical weakness enforce correctness

### 7. Representative Evidence Policy

Use one evidence row per synthesize slot.

Preferred source:

- slot-specific representative evidence from `sample_evidence_map`

Fallback:

- weakness-level representative evidence view built from a bounded local merge of at most a few rows

But the fallback merge must remain structural and short. It must not recreate a free-form narrative channel.

## Implementation Plan

### Step 1

Add a local synthesize-side builder such as:

- `build_synthesis_input_view(weakness, representative_evidence)`

### Step 2

Add deterministic `canonical_summary` generation helpers:

- one direct mapping for stable weaknesses
- one controlled bucketing helper for `logic_error`

### Step 3

Update `generate_spec()` payload construction to consume only the new input view.

### Step 4

Update synthesize prompt expectations so the model no longer sees `root_cause` and `capability_cliff`.

### Step 5

Confirm feasibility and fallback code still use representative evidence, not cluster narrative.

### Step 6

Add regression tests covering:

- payload no longer contains aggregate narrative fields
- `canonical_summary` is deterministic for each canonical weakness
- `logic_error` summary selection stays within allowed buckets
- synthesize still grounds `key_trap` in representative evidence

## Acceptance Criteria

The change is complete when all of the following are true:

1. synthesize model payload no longer contains:
   - `refined_root_cause`
   - `capability_cliff`
   - `misdiagnosed_ids`
   - `misdiagnosis_corrections`

2. synthesize payload does contain:
   - `fault_type`
   - primary `sub_tag`
   - `frequency`
   - `sample_task_ids`
   - deterministic `canonical_summary`
   - one representative evidence row

3. feasibility checks remain unchanged in responsibility:
   - they validate by weakness contract and evidence grounding
   - they do not use `canonical_summary`

4. on the same fixed `diagnoses.jsonl`, serial and concurrency-derived aggregate outputs may still differ in narrative text, but synthesize structural outputs become more stable:
   - attempted weaknesses stable
   - specs by weakness stable
   - fallback count variance reduced

## Risks

- if `canonical_summary` for `logic_error` is too coarse, specs may become generic
- if evidence selection is poor, synthesize may lose useful context even with a good contract
- if prompt text still implicitly assumes aggregate narrative, payload cleanup alone will not fully stabilize outputs

## Recommendation

Implement this as a minimal, local contract change in synthesize only.

Do not redesign upstream artifact schemas yet.

That keeps the change:

- low risk
- easy to verify
- aligned with current architecture
- directly targeted at the remaining stability issue
