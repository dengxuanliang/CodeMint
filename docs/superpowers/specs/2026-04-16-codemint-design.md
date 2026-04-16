# CodeMint Design Spec

> 大模型代码弱项挖掘与定向合成出题 Prompt Agent

## 1. Overview

CodeMint is a Python CLI tool that analyzes LLM code generation evaluation logs, identifies systematic weaknesses, and produces structured specs for targeted problem synthesis. The core loop: **diagnose → aggregate → synthesize**.

### Goals

1. Per-task structured root cause analysis of failed code generations
2. Top-N weakness extraction with cross-sample collective diagnosis
3. Transform weaknesses into structured specs consumable by downstream problem-generation agents

### Architecture

Four-stage file pipeline. Each stage communicates via JSONL files, supporting resumption (gap-fill, not overwrite) and independent re-runs. Model calls use OpenAI-compatible SDK via `base_url` + `api_key`.

```
codemint diagnose  → outputs/<run_id>/diagnoses.jsonl
codemint aggregate → outputs/<run_id>/weaknesses.json
codemint synthesize→ outputs/<run_id>/specs.jsonl
codemint run       → chains all three stages
```

Every run produces `outputs/<run_id>/run_metadata.json` for traceability.

---

## 2. Input Layer: LogLoader

### Abstraction

```python
class BaseLoader(ABC):
    @abstractmethod
    def load(self, paths: list[Path]) -> list[TaskRecord]: ...

class SplitFileLoader(BaseLoader): ...   # inference + results as separate files
class MergedFileLoader(BaseLoader): ...  # single merged file
```

Auto-detection: by file count and field presence. CLI accepts `-i` with any number of files.

### TaskRecord

```python
@dataclass
class TaskRecord:
    task_id: int
    content: str                # problem statement
    canonical_solution: str     # reference solution
    completion: str             # model-generated code
    test_code: str              # test harness
    labels: dict                # programming_language, category, difficulty, etc.
    accepted: bool              # pass@1 binary result
    metrics: dict               # additional metrics
    extra: dict                 # any other fields
```

---

## 3. Module 1: Diagnose

### Responsibility

Per-task structured root cause analysis. Outputs `diagnoses.jsonl`.

### Flow

```
Input tasks
  → Rule-based fast screening (12 built-in + user-defined rules)
  → Conflict resolution: priority-ordered, first-match-wins
  → Medium+ severity rule results → model secondary confirmation (can override rule)
  → R010 (AssertionError) and rule-miss → model deep analysis
  → Resumption: by task_id gap-fill, existing results preserved
  → Output diagnoses.jsonl
```

### Built-in Rules (with multi-language variants)

| ID   | Pattern                          | fault_type       | Priority |
|------|----------------------------------|------------------|----------|
| R007 | Timeout / TLE                    | modeling         | 1 (highest) |
| R006 | RecursionError / StackOverflow   | modeling         | 2        |
| R009 | Compilation error (C++/Java/Go)  | surface          | 3        |
| R001 | SyntaxError / IndentationError   | surface          | 4        |
| R002 | NameError: undefined variable    | surface          | 5        |
| R003 | ImportError / ModuleNotFoundError| surface          | 6        |
| R004 | TypeError: argument mismatch     | implementation   | 7        |
| R008 | Empty output / missing return    | implementation   | 8        |
| R012 | Division by zero / math domain   | edge_handling    | 9        |
| R005 | IndexError / KeyError            | edge_handling    | 10       |
| R011 | Output format mismatch           | surface          | 11       |
| R010 | AssertionError                   | → model analysis | 12       |

Each rule has multi-language variants (e.g., `IndexError` maps to Java's `ArrayIndexOutOfBoundsException`).

### Rule Misclassification Correction

Rules classify by surface error type, which may be misleading (e.g., IndexError caused by wrong algorithm, not missing boundary check). For medium+ severity rule results, the model performs secondary confirmation and can override the rule's classification. Cost is bounded (only medium+ subset).

### User-Configurable Rules

```yaml
rules:
  custom_patterns:
    - name: "custom_timeout"
      pattern: "TimeLimitExceeded|TLE"
      fault_type: "implementation"
      sub_tag: "time_complexity_exceeded"
      severity: "high"
  disabled_rules: ["R003"]
  severity_overrides:
    "missing_import": "low"
  rule_priority: [R007, R006, ...]  # custom priority order
```

Without a config file, built-in rules apply. With config, rules are merged.

### Two-Level Classification Taxonomy

**5 fixed orthogonal primary categories:**

| Category        | Description                                    |
|-----------------|------------------------------------------------|
| comprehension   | Misunderstood the problem statement            |
| modeling        | Wrong algorithm/approach selection              |
| implementation  | Correct approach, wrong code                   |
| edge_handling   | Missing boundary/special case handling          |
| surface         | Syntax, imports, formatting errors              |

**Extensible sub-tags:** Up to 20 per primary category. When exceeded, semantic merging is triggered (embedding cosine similarity > 0.85 → model confirms merge).

### Diagnosis Output Structure

```json
{
  "task_id": 101,
  "fault_type": "modeling",
  "sub_tags": ["chose_greedy_over_dp"],
  "severity": "high",
  "description": "Model used greedy approach on a problem requiring DP with overlapping subproblems",
  "evidence": {
    "wrong_line": "result = sorted(items, key=lambda x: x.value/x.weight, reverse=True)",
    "correct_approach": "2D DP with dp[i][w] = max(dp[i-1][w], dp[i-1][w-wi] + vi)",
    "failed_test": "input: n=50, items=[...]; expected: 120, got: 100"
  },
  "enriched_labels": {
    "algorithm_type": "dynamic_programming",
    "pattern": "0-1_knapsack_variant",
    "cognitive_difficulty": "requires_recognizing_non_greedy_optimal_substructure"
  },
  "confidence": 0.85,
  "diagnosis_source": "rule_confirmed_by_model | rule_only | model_deep",
  "prompt_version": "v2"
}
```

---

## 4. Module 2: Aggregate

### Responsibility

Cluster weaknesses → repair verification → collective diagnosis → causal chain analysis → ranked output. Outputs `weaknesses.json`.

### Flow

```
diagnoses.jsonl
  → Repair verification (3-level degradation, auto-detect)
      Level 1: Evaluation framework HTTP API execution verification (most accurate)
      Level 2: Cross-model verification (different model judges repair correctness)
      Level 3: Self-consistency check (model self-reviews repair logic)
      Auto-detects available level, prefers highest.
  → Verification failure → re-diagnose (max 1 retry)
  → Second failure → mark verification_status: "unverified", force confidence ≤ 0.5
  → Hard clustering by (fault_type, primary sub_tag)
  → Sub-tag normalization:
      Exact match → rule-based merge
      Semantic similarity (cosine > 0.85) → model confirms merge
      Merge mappings recorded in tag_mappings
  → Collective diagnosis (per cluster, up to 15 tasks sent together to model)
      Unverified diagnoses participate as low-weight samples
  → Causal chain DAG analysis (identify root-cause → downstream relationships)
  → Multi-dimensional parallel ranking (no composite score)
  → Incremental merge support: --previous weaknesses.json
  → Output weaknesses.json
```

### Repair Verification 3-Level Degradation

| Level | Method | When Available |
|-------|--------|----------------|
| 1 | Call evaluation framework HTTP API to execute repaired code | `evaluation_api.base_url` configured and reachable |
| 2 | Cross-model verification: send original + repaired code to a different model | Always (requires model API) |
| 3 | Self-consistency check: same model reviews its own repair logic | Always |

Auto-detection: probe Level 1 endpoint on startup; fall back to Level 2/3 if unreachable.

### Collective Diagnosis

After clustering, all tasks in the same weakness cluster (up to 15) are sent to the model in a single call for cross-sample deep analysis. This discovers:
- Refined root cause (sharper than per-task diagnosis)
- Capability cliff (precise boundary where model fails)
- Misdiagnosed tasks (reclassification if needed)
- Cluster coherence score

### Causal Chain Analysis

Identifies causal DAG among weaknesses. Example: "fails to recognize overlapping subproblems" (root) → "chooses greedy over DP" (downstream). Training the root cause is more efficient than training symptoms.

### Ranking Dimensions (parallel, no composite score)

- `by_frequency`: most common weaknesses
- `by_difficulty`: weaknesses concentrated in hard problems
- `by_trainability`: weaknesses most amenable to training improvement

### weaknesses.json Output Structure

```json
{
  "weaknesses": [
    {
      "rank": 1,
      "fault_type": "modeling",
      "sub_tags": ["chose_greedy_over_dp"],
      "frequency": 45,
      "sample_task_ids": [101, 203, 307],
      "trainability": 0.9,
      "collective_diagnosis": {
        "refined_root_cause": "Cannot identify overlapping subproblems; defaults to greedy when problem structure is ambiguous",
        "capability_cliff": "Correct on n≤20 where greedy happens to work; fails systematically on n>20",
        "misdiagnosed_ids": [405],
        "misdiagnosis_corrections": {
          "405": "Actually comprehension error, not modeling"
        },
        "cluster_coherence": 0.88
      }
    }
  ],
  "rankings": {
    "by_frequency": [1, 3, 2],
    "by_difficulty": [2, 1, 3],
    "by_trainability": [1, 2, 3]
  },
  "causal_chains": [
    {
      "root": "fails_to_identify_overlapping_subproblems",
      "downstream": ["chose_greedy_over_dp", "missed_memoization"],
      "training_priority": "Train root cause first; downstream symptoms may self-resolve"
    }
  ],
  "tag_mappings": {
    "greedy_instead_of_dp": "chose_greedy_over_dp",
    "wrong_greedy": "chose_greedy_over_dp"
  }
}
```

---

## 5. Module 3: Synthesize

### Responsibility

Transform weaknesses into structured specs for downstream problem-generation agents. Outputs `specs.jsonl`. Does NOT produce complete problems — only specs.

### Flow

```
weaknesses.json (Top-N)
  → Allocate spec count per weakness
      base = specs_per_weakness (default 3)
      causal root +2
      trainability ≥ 0.8 +1
      cap at max_per_weakness (default 8)
  → Pre-assign diversity_tags combination + difficulty alternation
  → Generate spec per slot (model call)
      Prompt includes:
        - Weakness root_cause + capability_cliff
        - Original failed task evidence (wrong_line + correct_approach, up to 3 tasks)
        - Pre-assigned diversity_tags combination
        - must_avoid constraints (existing spec patterns)
  → Feasibility self-check (model writes pseudocode skeleton; failure → reject & regenerate)
  → De-duplication check
      Compare diversity_tags against existing specs
      >50% overlap (2 of 3 dimensions identical) → reject, re-assign combination
      Intra-weakness specs also cross-checked
      Max 2 regeneration attempts
  → Output specs.jsonl
```

### De-Duplication: Orthogonal Dimension Matrix (Pre-Assigned)

Three dimensions form the problem space. Combinations are pre-assigned before generation, not checked post-hoc.

| Dimension        | Values |
|------------------|--------|
| narrative_theme  | Two-level: fixed `generic` list + `domain_adaptive` (model generates when generic doesn't fit; must not repeat within same weakness) |
| data_structure   | array, tree, graph, string, matrix, heap, stack, hash_map (user-extensible) |
| constraint_scale | small (n≤100), medium (n≤10⁴), large (n≤10⁶) |

`narrative_theme` generic list and `data_structure` values are user-configurable and extensible via `codemint.yaml`. Specific enum values to be aligned with established benchmark taxonomies (pending user confirmation on reference source).

### Difficulty Configuration

```yaml
synthesize:
  difficulty_levels: ["medium", "hard"]  # easy not allowed; user-configurable
  difficulty_distribution: "balanced"     # balanced = even split / weighted_hard = hard 2/3
```

Validation: if `easy` appears in config, reject with error.

### key_trap Constraint

Must be derived from original failed task evidence (`evidence.wrong_line` + `evidence.correct_approach`). Generic descriptions not allowed. This is the soul of each spec — it encodes the precise trap that the evaluated model fell into.

### Spec Output Structure

```json
{
  "spec_id": "modeling__chose_greedy_over_dp__01",
  "target_weakness": {
    "fault_type": "modeling",
    "sub_tags": ["chose_greedy_over_dp"],
    "root_cause": "Cannot identify optimal substructure; defaults to greedy",
    "capability_cliff": "Greedy diverges from optimal when n>20"
  },
  "problem_spec": {
    "algorithm_type": "dynamic_programming",
    "difficulty": "hard",
    "narrative_theme": "logistics_scheduling",
    "constraints": {
      "n_range": [1, 100000],
      "value_range": [1, 1000000000],
      "time_limit": "2s",
      "memory_limit": "256MB"
    },
    "key_trap": "Greedy produces correct answer for first 3 examples; fails at n≥50 where local optimum ≠ global optimum",
    "must_cover": [
      "2D state transition: position × remaining capacity",
      "Must recognize overlapping subproblems, not independent",
      "Optimal solution requires backtracking, not local choice"
    ],
    "must_avoid": [
      "Must not be classic 0-1 knapsack or its direct variant",
      "Problem statement must not contain DP/dynamic programming/knapsack hints",
      "Must not be solvable by sort+greedy"
    ]
  },
  "verification_spec": {
    "min_test_cases": 10,
    "must_include_edge_cases": [
      "n=1 single element",
      "All items equal value (greedy happens to be correct)",
      "n=50 smallest scale where greedy first fails",
      "n=100000 max scale stress test"
    ],
    "brute_force_verifiable": true,
    "brute_force_complexity_limit": "O(2^n), n<=20"
  },
  "diversity_tags": {
    "narrative_theme": "logistics_scheduling",
    "data_structure": "array",
    "constraint_scale": "large"
  },
  "generation_hints": {
    "solution_approach": "2D DP: dp[i][w] = max profit for first i items with capacity w",
    "common_wrong_approach": "Sort by value/weight ratio, greedily pick; ignores indivisibility",
    "distinguishing_test": "Construct 3 items: greedy picks A+B for 100, DP picks C for 120"
  },
  "language_constraint": {
    "target_languages": ["python", "cpp"],
    "language_specific": false
  },
  "prompt_version": "v1"
}
```

### Downstream Usage

The spec is a precise requirements document for problem-generation agents. Each field maps to a section of the generation prompt:

| Spec Field | Downstream Usage |
|---|---|
| `target_weakness` | **Why** this problem exists — ensures it hits the right weakness |
| `key_trap` | **How** to design the trap — the soul of the problem |
| `must_cover / must_avoid` | Positive/negative constraints — prevents trivial or duplicate problems |
| `generation_hints` | Correct vs wrong approach contrast — ensures discriminative power |
| `verification_spec` | Test case quality standards |
| `diversity_tags` | Problem skin/packaging (scenario, data structure) |
| `language_constraint` | Which language(s) the standard solution should target |

### Incremental Mode

```bash
codemint synthesize -i weaknesses.json --existing specs_v1.jsonl -o specs_v2.jsonl
```

`--existing` loads prior specs; new specs avoid their diversity_tags combinations.

---

## 6. Module 4: Run

### Responsibility

Chain diagnose → aggregate → synthesize in a single command.

### Selective Stage Skipping

- Detect intermediate artifacts and check completeness
- Complete (missing count = 0) → skip stage
- Incomplete (missing count > 0) → gap-fill missing records only
- CLI override: `--from aggregate` forces start from a specific stage

### Dry-Run Mode

```bash
codemint run -i eval_log.jsonl --dry-run
```

Output:
```
Estimated model calls: 1023 (diagnose: 800, aggregate: 187, synthesize: 36)
Estimated tokens: ~2.1M input, ~0.5M output
Rule-screened (no model call): 200/1000
Estimated time: ~25min (concurrency=5)
```

No actual model calls made. Statistics and estimation only.

---

## 7. Production Infrastructure

### Model Call Protection

```yaml
model:
  base_url: "..."
  api_key: "..."
  analysis_model: "claude-sonnet-4-20250514"
  evaluated_model: "gpt-4"             # logged; same-model warning
  max_concurrency: 5
  max_retries: 3
  retry_backoff: "exponential"         # 1s, 2s, 4s
  timeout: 120                         # seconds
  max_input_tokens: 8000               # truncation priority: test_code → solution → content (never)
```

**Same-model blind spot warning:** When `analysis_model == evaluated_model`, output includes `"self_analysis_warning": true` and summary warns of potential systematic bias.

### Model Output Parsing Protection

- Every model output validated against JSON Schema
- Parse failure → auto-retry once (with format error hint in prompt)
- Second failure → log to `errors.jsonl`, skip record, do not block pipeline

### API Fault Tolerance

- Exponential backoff retry (1s, 2s, 4s), up to `max_retries`
- Unrecoverable errors → `errors.jsonl`, do not block pipeline

### Token Budget Management

Per model call, estimate token count. When exceeding `max_input_tokens`, truncate by priority:
1. `test_code` → keep first N test cases
2. `canonical_solution` → keep first 200 lines
3. `content` → never truncate

### Progress Observability

```
[diagnose] ████████░░░░ 673/1000 (67%) | 12 errors | ETA 3min
```

End-of-run summary: processed / skipped / errors / elapsed time.

### Prompt Version Management

```
prompts/
  diagnose_deep_analysis.txt
  diagnose_rule_confirm.txt
  aggregate_collective_diagnosis.txt
  aggregate_causal_chain.txt
  synthesize_spec_generation.txt
  synthesize_feasibility_check.txt
```

- Each prompt file has a version header
- Every output record includes `prompt_version` field
- Users can override default templates with custom prompts

### Run Traceability

```json
{
  "run_id": "20260416_143022_a3f1",
  "timestamp": "2026-04-16T14:30:22Z",
  "config_snapshot": {},
  "analysis_model": "claude-sonnet-4-20250514",
  "prompt_versions": { "diagnose": "v2", "aggregate": "v3", "synthesize": "v1" },
  "input_files": ["eval_log.jsonl"],
  "input_count": 1000,
  "stages_executed": ["diagnose", "aggregate", "synthesize"],
  "summary": {
    "diagnosed": 1000,
    "rule_screened": 200,
    "model_analyzed": 800,
    "errors": 3,
    "weaknesses_found": 12,
    "specs_generated": 36
  }
}
```

All output files stored under `outputs/<run_id>/`, never overwriting previous runs.

---

## 8. Global Configuration: codemint.yaml

```yaml
model:
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  analysis_model: "claude-sonnet-4-20250514"
  evaluated_model: "gpt-4"
  max_concurrency: 5
  max_retries: 3
  retry_backoff: "exponential"
  timeout: 120
  max_input_tokens: 8000

evaluation_api:
  base_url: "http://localhost:8080"    # optional

rules:
  custom_patterns: []
  disabled_rules: []
  severity_overrides: {}
  rule_priority: []                    # empty = use default priority

aggregate:
  verification_level: "auto"           # auto / exec / cross-model / self-check
  max_cluster_size: 15
  sub_tag_limit_per_category: 20

synthesize:
  specs_per_weakness: 3
  max_per_weakness: 8
  top_n: 10
  difficulty_levels: ["medium", "hard"]
  difficulty_distribution: "balanced"
  diversity_overlap_threshold: 0.5
  max_regeneration_attempts: 2
  narrative_themes:
    generic: []                        # to be aligned with benchmark taxonomy
    domain_adaptive: true
  data_structures:
    - array
    - tree
    - graph
    - string
    - matrix
    - heap
    - stack
    - hash_map
```

---

## 9. Design Decisions Log

| Decision | Rationale |
|---|---|
| File pipeline over in-memory | Resumption, independent re-runs, inspectable intermediate results |
| Rule + model hybrid analysis | Rules are fast/cheap for surface errors; model reserved for reasoning-required cases |
| Two-level taxonomy (5 fixed + sub-tags) | Orthogonality + extensibility + controlled growth |
| Multi-dimensional ranking without composite score | Avoids arbitrary weighting; each dimension has independent value |
| Collective diagnosis | Cross-sample analysis discovers patterns invisible to per-task diagnosis |
| Causal chain DAG | Training root causes is more efficient than training symptoms |
| 3-level verification degradation | Adapts to user's infrastructure availability |
| Pre-assigned diversity tags | Prevents post-hoc clustering into similar problems |
| Spec output (not complete problems) | Clean separation of concerns; downstream agent handles generation |
| Gap-fill resumption | Preserves completed work on interruption |
| Prompt files with version tracking | Core asset management; enables A/B comparison |

---

## 10. Pending Items

- [ ] `narrative_theme` generic list: align with established benchmark taxonomy (user to confirm reference source)
- [ ] `data_structure` list: may need expansion based on evaluation set coverage
