from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from codemint.cli import app
from codemint.config import CodeMintConfig
from codemint.io.jsonl import append_jsonl, read_jsonl
from codemint.models.diagnosis import DiagnosisEvidence, DiagnosisRecord
from codemint.models.spec import (
    DiversityTags,
    GenerationHints,
    LanguageConstraint,
    ProblemConstraints,
    ProblemSpec,
    SpecRecord,
    TargetWeakness,
    VerificationSpec,
)
from codemint.models.weakness import CausalChain, CollectiveDiagnosis, RankingSet, WeaknessEntry, WeaknessReport


def test_canonicalized_weakness_keys_pull_real_evidence_from_diagnoses() -> None:
    from codemint.synthesize.pipeline import build_original_evidence_map

    report = WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="modeling",
                sub_tags=["state_tracking"],
                frequency=5,
                sample_task_ids=[1, 2, 3],
                trainability=0.9,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="State invariants break after transitions.",
                    capability_cliff="Long chains require carrying latent state.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.94,
                ),
            )
        ],
        rankings=RankingSet(by_frequency=[1], by_difficulty=[1], by_trainability=[1]),
        causal_chains=[CausalChain(root="state_tracking", downstream=[], training_priority="high")],
        tag_mappings={"carry_state": "state_tracking", "state_tracking": "state_tracking"},
    )
    diagnosis = DiagnosisRecord(
        task_id=11,
        fault_type="modeling",
        sub_tags=["carry_state"],
        severity="high",
        description="State is lost at the final transition.",
        evidence=DiagnosisEvidence(
            wrong_line="Dropped the carry before processing the final warehouse.",
            correct_approach="Preserve carry state through the final transition.",
            failed_test="Final warehouse requires carrying state into the last step.",
        ),
        enriched_labels={},
        confidence=0.9,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )

    evidence_map = build_original_evidence_map(report, [diagnosis])

    assert evidence_map["state_tracking"]["wrong_line"] == diagnosis.evidence.wrong_line
    assert evidence_map["state_tracking"]["correct_approach"] == diagnosis.evidence.correct_approach


def test_build_original_evidence_map_falls_back_to_sample_task_ids_when_tags_do_not_match() -> None:
    from codemint.synthesize.pipeline import build_original_evidence_map

    report = WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="implementation",
                sub_tags=["missing_code_block"],
                frequency=1,
                sample_task_ids=[101],
                trainability=0.6,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Executable code is missing.",
                    capability_cliff="The model emits explanation instead of code.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.9,
                ),
            )
        ],
        rankings=RankingSet(by_frequency=[1], by_difficulty=[1], by_trainability=[1]),
        causal_chains=[CausalChain(root="missing_code_block", downstream=[], training_priority="high")],
        tag_mappings={"deep_analysis": "deep_analysis"},
    )
    diagnosis = DiagnosisRecord(
        task_id=101,
        fault_type="implementation",
        sub_tags=["deep_analysis"],
        severity="medium",
        description="Deep analysis diagnosis",
        evidence=DiagnosisEvidence(
            wrong_line="I would solve this by defining a function, but the final code is missing.",
            correct_approach="Return executable solve(x) code directly.",
            failed_test="assert solve(1) == 2",
        ),
        enriched_labels={},
        confidence=0.8,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )

    evidence_map = build_original_evidence_map(report, [diagnosis])

    assert evidence_map["missing_code_block"]["wrong_line"] == diagnosis.evidence.wrong_line
    assert evidence_map["missing_code_block"]["correct_approach"] == diagnosis.evidence.correct_approach


def test_build_original_evidence_map_prefers_function_name_mismatch_sample_evidence() -> None:
    from codemint.synthesize.pipeline import build_original_evidence_map

    report = WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="implementation",
                sub_tags=["function_name_mismatch"],
                frequency=1,
                sample_task_ids=[103],
                trainability=0.6,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Wrong public entry point is exposed.",
                    capability_cliff="Strict harnesses require the exact solve name.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.95,
                ),
            )
        ],
        rankings=RankingSet(by_frequency=[1], by_difficulty=[1], by_trainability=[1]),
        causal_chains=[CausalChain(root="function_name_mismatch", downstream=[], training_priority="high")],
        tag_mappings={"deep_analysis": "function_name_mismatch", "function_name_mismatch": "function_name_mismatch"},
    )
    diagnosis = DiagnosisRecord(
        task_id=103,
        fault_type="implementation",
        sub_tags=["deep_analysis"],
        severity="medium",
        description="The implementation exposes solve_value instead of solve.",
        evidence=DiagnosisEvidence(
            wrong_line="```python\ndef solve_value(x):\n    return x * 2\n```",
            correct_approach="Define the exact solve(x) entry point expected by the harness.",
            failed_test="NameError: name 'solve' is not defined",
        ),
        enriched_labels={},
        confidence=0.8,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )

    evidence_map = build_original_evidence_map(report, [diagnosis])

    assert "solve_value" in evidence_map["function_name_mismatch"]["wrong_line"]
    assert "solve(x)" in evidence_map["function_name_mismatch"]["correct_approach"]


def test_synthesize_command_reads_real_evidence_from_diagnoses_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "demo-run"
    run_dir.mkdir(parents=True)
    (run_dir / "weaknesses.json").write_text(_report().model_dump_json(), encoding="utf-8")
    (run_dir / "diagnoses.jsonl").write_text(_diagnosis().model_dump_json() + "\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "synthesize",
            "--output-root",
            str(tmp_path / "artifacts"),
            "--run-id",
            "demo-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (run_dir / "specs.jsonl").exists()
    specs_text = (run_dir / "specs.jsonl").read_text(encoding="utf-8")
    assert "Dropped the carry before processing the final warehouse." in specs_text
    assert "Wrote 6 synthesized specs" in result.stdout


def test_synthesize_command_supports_incremental_mode_with_existing_specs(tmp_path: Path) -> None:
    weaknesses_path = tmp_path / "weaknesses.json"
    output_path = tmp_path / "specs_v2.jsonl"
    existing_path = tmp_path / "specs_v1.jsonl"
    weaknesses_path.write_text(_report().model_dump_json(), encoding="utf-8")
    append_jsonl(existing_path, [_existing_spec().model_dump(mode="json")])

    result = CliRunner().invoke(
        app,
        [
            "synthesize",
            "-i",
            str(weaknesses_path),
            "--existing",
            str(existing_path),
            "-o",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 6
    assert "Wrote" in result.stdout


def test_synthesize_logs_generation_failures_and_continues(tmp_path: Path) -> None:
    from codemint.synthesize.pipeline import run_synthesize

    output_path = tmp_path / "specs.jsonl"
    calls = {"count": 0}

    def invoke_model(payload: dict) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("generation failed")
        return {
            "algorithm_type": "dynamic programming",
            "difficulty": "hard",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": "Reference the original evidence: Dropped the carry before processing the final warehouse.",
            "must_cover": ["state_tracking", "Preserve carry state through the final transition."],
            "must_avoid": ["verbatim reuse of prior tasks"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["Final warehouse requires carrying state into the last step."],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Preserve carry state through the final transition.",
                "common_wrong_approach": "Dropped the carry before processing the final warehouse.",
                "distinguishing_test": "Final warehouse requires carrying state into the last step.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        }

    specs = run_synthesize(
        _report(),
        output_path,
        config=CodeMintConfig.model_validate(
            {
                "synthesize": {
                    "specs_per_weakness": 2,
                    "max_per_weakness": 2,
                    "top_n": 1,
                    "narrative_themes": {
                        "generic": ["warehouses", "ports"],
                        "domain_adaptive": False,
                    },
                    "data_structures": ["array"],
                }
            }
        ),
        original_evidence={
            "state_tracking": {
                "wrong_line": "Dropped the carry before processing the final warehouse.",
                "correct_approach": "Preserve carry state through the final transition.",
                "failed_test": "Final warehouse requires carrying state into the last step.",
            }
        },
        invoke_model=invoke_model,
    )

    assert len(specs) == 1
    logged = read_jsonl(tmp_path / "errors.jsonl")
    assert len(logged) == 1
    assert logged[0]["stage"] == "synthesize"
    assert logged[0]["error_type"] == "spec_generation_failed"


def test_synthesize_raises_when_all_specs_fail(tmp_path: Path) -> None:
    from codemint.synthesize.pipeline import run_synthesize

    output_path = tmp_path / "specs.jsonl"

    with pytest.raises(ValueError, match="No specs were synthesized successfully"):
        run_synthesize(
            _report(),
            output_path,
            config=CodeMintConfig.model_validate(
                {
                    "synthesize": {
                        "specs_per_weakness": 1,
                        "max_per_weakness": 1,
                        "top_n": 1,
                        "narrative_themes": {"generic": ["warehouses"], "domain_adaptive": False},
                        "data_structures": ["array"],
                    }
                }
            ),
            invoke_model=lambda payload: (_ for _ in ()).throw(RuntimeError("boom")),
        )


def test_synthesize_uses_function_name_mismatch_fallback_when_generation_fails(tmp_path: Path) -> None:
    from codemint.synthesize.pipeline import run_synthesize

    report = WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="implementation",
                sub_tags=["function_name_mismatch"],
                frequency=6,
                sample_task_ids=[2201, 2202, 2203],
                trainability=0.6,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="Function definition name does not match the expected callable entry point.",
                    capability_cliff="Breaks when the harness requires the exact solve(x) name.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=1.0,
                ),
            )
        ],
        rankings=RankingSet(by_frequency=[1], by_difficulty=[1], by_trainability=[1]),
        causal_chains=[CausalChain(root="function_name_mismatch", downstream=[], training_priority="high")],
        tag_mappings={"function_name_mismatch": "function_name_mismatch"},
    )
    diagnoses = [
        DiagnosisRecord(
            task_id=2201,
            fault_type="implementation",
            sub_tags=["function_name_mismatch"],
            severity="high",
            description="Wrong function name.",
            evidence=DiagnosisEvidence(
                wrong_line="def solve_value(x):\n    return x + 3",
                correct_approach="Define the exact solve(x) entry point expected by the harness.",
                failed_test="NameError: name 'solve' is not defined",
            ),
            enriched_labels={},
            confidence=1.0,
            diagnosis_source="model_deep",
            prompt_version="v1",
        )
    ]

    specs = run_synthesize(
        report,
        tmp_path / "specs.jsonl",
        config=CodeMintConfig.model_validate(
            {
                "synthesize": {
                    "top_n": 1,
                    "specs_per_weakness": 1,
                    "max_per_weakness": 1,
                    "narrative_themes": {"generic": ["warehouses"], "domain_adaptive": False},
                    "data_structures": ["array"],
                }
            }
        ),
        diagnoses=diagnoses,
        invoke_model=lambda payload: (_ for _ in ()).throw(ValueError("forced generation failure")),
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.target_weakness.sub_tags == ["function_name_mismatch"]
    assert any("exact callable entry point" in item.lower() for item in spec.problem_spec.must_cover)
    assert any("alternate public function names" in item.lower() for item in spec.problem_spec.must_avoid)


def test_missing_code_block_local_feasibility_rejects_spec_without_output_constraints() -> None:
    from codemint.synthesize.feasibility import check_feasibility

    spec = SpecRecord(
        spec_id="spec-0099",
        target_weakness=TargetWeakness(
            fault_type="implementation",
            sub_tags=["missing_code_block"],
            root_cause="Explanation is returned instead of code.",
            capability_cliff="Code emission fails under direct implementation prompts.",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="simulation",
            difficulty="medium",
            narrative_theme="warehouses",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="The trap fails when the solver explains the approach instead of returning a callable solve function.",
            must_cover=["implementation correctness"],
            must_avoid=["duplicate prompt wording"],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single value input"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        generation_hints=GenerationHints(
            solution_approach="Return the implementation directly.",
            common_wrong_approach="Explain the intended code without emitting it.",
            distinguishing_test="Check that a callable solve function exists.",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )

    result = check_feasibility(
        spec,
        original_evidence={
            "wrong_line": "I would solve this by defining a function and returning x plus one, but the final code block is missing.",
            "correct_approach": "Generate the Python function definition def solve(x): return x + 1.",
            "failed_test": "Execution failed because no callable solve function was produced.",
        },
    )

    assert result.accepted is False
    assert "executable code output" in result.reason


def test_function_name_mismatch_local_feasibility_rejects_spec_without_exact_entry_point() -> None:
    from codemint.synthesize.feasibility import check_feasibility

    spec = SpecRecord(
        spec_id="spec-0100",
        target_weakness=TargetWeakness(
            fault_type="implementation",
            sub_tags=["function_name_mismatch"],
            root_cause="Function name mismatches break execution.",
            capability_cliff="Strict harnesses require an exact entry point.",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="simulation",
            difficulty="medium",
            narrative_theme="warehouses",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="The trap fails when the harness cannot call `solve` because the original evidence exposed `solve_value`.",
            must_cover=["implementation correctness", "public interface consistency"],
            must_avoid=["duplicate prompt wording"],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single value input"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        generation_hints=GenerationHints(
            solution_approach="Expose the expected callable.",
            common_wrong_approach="Expose a differently named helper instead.",
            distinguishing_test="Call solve() directly from the harness.",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )

    result = check_feasibility(
        spec,
        original_evidence={
            "wrong_line": "def solve_value(x):",
            "correct_approach": "Define the exact solve(x) entry point expected by the harness.",
            "failed_test": "NameError: name 'solve' is not defined",
        },
    )

    assert result.accepted is False
    assert "exact public entry-point" in result.reason


def test_function_name_mismatch_local_feasibility_accepts_semantic_equivalent_contract_language() -> None:
    from codemint.synthesize.feasibility import check_feasibility

    spec = SpecRecord(
        spec_id="spec-0100a",
        target_weakness=TargetWeakness(
            fault_type="implementation",
            sub_tags=["function_name_mismatch"],
            root_cause="Function name mismatches break execution.",
            capability_cliff="Strict harnesses require an exact entry point.",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="simulation",
            difficulty="medium",
            narrative_theme="warehouses",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="The trap fails when the harness cannot call `solve` because the original evidence exposed `solve_value`.",
            must_cover=[
                "Require one public function named solve for the harness entry point.",
                "The checker must call the exact solver entrypoint directly.",
            ],
            must_avoid=[
                "Forbid helper entrypoint aliases like solve_value or solver.",
                "Do not rename the public callable.",
            ],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single value input"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        generation_hints=GenerationHints(
            solution_approach="Expose a single public function named solve.",
            common_wrong_approach="Expose helper aliases like solve_value or solver.",
            distinguishing_test="Call solve() directly from the harness.",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )

    result = check_feasibility(
        spec,
        original_evidence={
            "wrong_line": "def solve_value(x):",
            "correct_approach": "Define the exact solve(x) entry point expected by the harness.",
            "failed_test": "NameError: name 'solve' is not defined",
        },
    )

    assert result.accepted is True


def test_markdown_formatting_local_feasibility_rejects_spec_without_raw_output_rules() -> None:
    from codemint.synthesize.feasibility import check_feasibility

    spec = SpecRecord(
        spec_id="spec-0101",
        target_weakness=TargetWeakness(
            fault_type="surface",
            sub_tags=["markdown_formatting"],
            root_cause="Markdown fences wrap otherwise executable code.",
            capability_cliff="Raw-output consumers fail when fence markers are present.",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="simulation",
            difficulty="medium",
            narrative_theme="warehouses",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="The trap reproduces the original ```python fenced solve(x) output instead of raw executable code.",
            must_cover=["implementation correctness", "python function output"],
            must_avoid=["duplicate prompt wording"],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single value input"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        generation_hints=GenerationHints(
            solution_approach="Return the raw implementation directly.",
            common_wrong_approach="Wrap the answer in markdown fences.",
            distinguishing_test="Reject answers containing ``` or stray backticks.",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )

    result = check_feasibility(
        spec,
        original_evidence={
            "wrong_line": "```python\ndef solve(x):\n    return x + 1\n```",
            "correct_approach": "Return raw executable code without markdown fences.",
            "failed_test": "SyntaxError: invalid syntax",
        },
    )

    assert result.accepted is False
    assert "raw executable output" in result.reason


def test_markdown_formatting_local_feasibility_accepts_semantic_equivalent_contract_language() -> None:
    from codemint.synthesize.feasibility import check_feasibility

    spec = SpecRecord(
        spec_id="spec-0101a",
        target_weakness=TargetWeakness(
            fault_type="surface",
            sub_tags=["markdown_formatting"],
            root_cause="Markdown fences wrap otherwise executable code.",
            capability_cliff="Raw-output consumers fail when fence markers are present.",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="simulation",
            difficulty="medium",
            narrative_theme="warehouses",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="The trap reproduces the original ```python fenced solve(x) output instead of raw executable code.",
            must_cover=[
                "Return plain executable program text only.",
                "The harness expects raw code output.",
            ],
            must_avoid=[
                "Do not wrap answers in markdown code fences.",
                "Avoid backticks or formatting delimiters around the final code.",
            ],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single value input"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        generation_hints=GenerationHints(
            solution_approach="Return the raw implementation directly.",
            common_wrong_approach="Wrap the answer in markdown fences.",
            distinguishing_test="Reject answers containing ``` or stray backticks.",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )

    result = check_feasibility(
        spec,
        original_evidence={
            "wrong_line": "```python\ndef solve(x):\n    return x + 1\n```",
            "correct_approach": "Return raw executable code without markdown fences.",
            "failed_test": "SyntaxError: invalid syntax",
        },
    )

    assert result.accepted is True


def test_syntax_error_local_feasibility_rejects_spec_without_syntactic_completeness_rules() -> None:
    from codemint.synthesize.feasibility import check_feasibility

    spec = SpecRecord(
        spec_id="spec-0102",
        target_weakness=TargetWeakness(
            fault_type="implementation",
            sub_tags=["syntax_error"],
            root_cause="The generated code is syntactically incomplete.",
            capability_cliff="Execution fails before semantic correctness is even tested.",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="simulation",
            difficulty="medium",
            narrative_theme="warehouses",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="The trap reproduces the original incomplete `def solve(x)` output without a valid body.",
            must_cover=["implementation correctness"],
            must_avoid=["duplicate prompt wording"],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single value input"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        generation_hints=GenerationHints(
            solution_approach="Return the implementation directly.",
            common_wrong_approach="Emit incomplete code that cannot parse.",
            distinguishing_test="Reject code with missing colons or missing bodies.",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )

    result = check_feasibility(
        spec,
        original_evidence={
            "wrong_line": "def solve(x)",
            "correct_approach": "Return syntactically complete executable code with a valid solve(x) definition.",
            "failed_test": "SyntaxError: expected ':'",
        },
    )

    assert result.accepted is False
    assert "syntactically complete executable code" in result.reason


def test_syntax_error_local_feasibility_accepts_semantic_equivalent_contract_language() -> None:
    from codemint.synthesize.feasibility import check_feasibility

    spec = SpecRecord(
        spec_id="spec-0102a",
        target_weakness=TargetWeakness(
            fault_type="implementation",
            sub_tags=["syntax_error"],
            root_cause="The generated code is syntactically incomplete.",
            capability_cliff="Execution fails before semantic correctness is even tested.",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="simulation",
            difficulty="medium",
            narrative_theme="warehouses",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="The trap reproduces the original incomplete `def solve(x)` output without a valid body.",
            must_cover=[
                "Require parseable and syntactically valid executable code.",
                "The emitted function definition must be complete.",
            ],
            must_avoid=[
                "Do not omit required punctuation like colons.",
                "Avoid malformed or partial function headers.",
            ],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single value input"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        generation_hints=GenerationHints(
            solution_approach="Return the implementation directly.",
            common_wrong_approach="Emit incomplete code that cannot parse.",
            distinguishing_test="Reject code with missing colons or missing bodies.",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )

    result = check_feasibility(
        spec,
        original_evidence={
            "wrong_line": "def solve(x)",
            "correct_approach": "Return syntactically complete executable code with a valid solve(x) definition.",
            "failed_test": "SyntaxError: expected ':'",
        },
    )

    assert result.accepted is True


def test_non_executable_code_local_feasibility_rejects_spec_without_output_presence_rules() -> None:
    from codemint.synthesize.feasibility import check_feasibility

    spec = SpecRecord(
        spec_id="spec-0103",
        target_weakness=TargetWeakness(
            fault_type="implementation",
            sub_tags=["non_executable_code"],
            root_cause="The response is not executable code.",
            capability_cliff="The model produces output that cannot be run directly.",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="simulation",
            difficulty="medium",
            narrative_theme="warehouses",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="The trap reproduces the original explanation-only response instead of executable solve(x) code.",
            must_cover=["implementation correctness"],
            must_avoid=["duplicate prompt wording"],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single value input"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        generation_hints=GenerationHints(
            solution_approach="Emit executable code directly.",
            common_wrong_approach="Emit a non-executable explanation.",
            distinguishing_test="Check that executable solve(x) code is present.",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )

    result = check_feasibility(
        spec,
        original_evidence={
            "wrong_line": "I would solve this by defining a function and returning x plus one.",
            "correct_approach": "Return executable solve(x) code directly.",
            "failed_test": "Execution failed because no runnable code was produced.",
        },
    )

    assert result.accepted is False
    assert "executable output presence" in result.reason


def test_generate_with_regeneration_passes_feasibility_feedback_into_retry() -> None:
    from codemint.synthesize.feasibility import FeasibilityResult
    from codemint.synthesize.pipeline import _generate_with_regeneration

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["missing_code_block"],
        frequency=1,
        sample_task_ids=[101],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Explanation is returned instead of code.",
            capability_cliff="Code emission fails under direct implementation prompts.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.95,
        ),
    )
    seen_constraints: list[list[str]] = []

    def invoke_model(payload: dict) -> dict:
        seen_constraints.append(list(payload.get("must_avoid_constraints", [])))
        return {
            "algorithm_type": "simulation",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": "The trap fails when the solver explains the approach instead of returning executable code for solve.",
            "must_cover": ["Executable code output with a callable solve function"],
            "must_avoid": ["Explanation-only or prose-only final answers"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Return the implementation directly.",
                "common_wrong_approach": "Explain the intended code without emitting it.",
                "distinguishing_test": "Check that a callable solve function exists.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        }

    feasibility_calls = {"count": 0}

    def feasibility_check(payload: dict) -> dict:
        feasibility_calls["count"] += 1
        if feasibility_calls["count"] == 1:
            return {"accepted": False, "reason": "Need stronger executable code output requirement."}
        return {"accepted": True, "reason": "ok"}

    spec = _generate_with_regeneration(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        spec_index=1,
        difficulty="medium",
        invoke_model=invoke_model,
        feasibility_check=feasibility_check,
        original_evidence={
            "wrong_line": "I would solve this by defining a function and returning x plus one, but the final code block is missing.",
            "correct_approach": "Generate executable solve code directly.",
            "failed_test": "Execution failed because no callable solve function was produced.",
        },
        overlap_threshold=0.5,
        existing_specs=[],
        max_attempts=2,
    )

    assert spec.problem_spec.must_cover
    assert len(seen_constraints) == 2
    assert any("Need stronger executable code output requirement." in item for item in seen_constraints[1])


def test_generate_with_regeneration_includes_structured_repair_mode_for_contract_mismatch() -> None:
    from codemint.synthesize.feasibility import FeasibilityResult
    from codemint.synthesize.pipeline import _generate_with_regeneration

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["function_name_mismatch"],
        frequency=1,
        sample_task_ids=[201],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Function name mismatches break execution.",
            capability_cliff="Strict harnesses require the exact public entry point.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.94,
        ),
    )
    seen_payloads: list[dict] = []

    def invoke_model(payload: dict) -> dict:
        seen_payloads.append(payload)
        return {
            "algorithm_type": "simulation",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": "The trap fails when `solve_value` is exposed instead of the required `solve` entry point.",
            "must_cover": ["exact callable entry point solve(x)"],
            "must_avoid": ["alternate public function names"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Implement the exact solve(x) entry point.",
                "common_wrong_approach": "Expose solve_value(x) instead of solve(x).",
                "distinguishing_test": "Call solve() directly from the harness.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        }

    feasibility_calls = {"count": 0}

    def feasibility_check(payload: dict) -> FeasibilityResult:
        feasibility_calls["count"] += 1
        if feasibility_calls["count"] == 1:
            return FeasibilityResult(
                accepted=False,
                reason="Function-name mismatch weakness spec must require a single exact public entry-point contract and forbid alternate public function names.",
                missing_contracts=[
                    "requires_exact_public_entry_point",
                    "forbids_alternate_public_names",
                ],
            )
        return FeasibilityResult(accepted=True, reason="ok")

    _generate_with_regeneration(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        spec_index=1,
        difficulty="medium",
        invoke_model=invoke_model,
        feasibility_check=feasibility_check,
        original_evidence={
            "wrong_line": "def solve_value(x):",
            "correct_approach": "Define the exact solve(x) entry point expected by the harness.",
            "failed_test": "NameError: name 'solve' is not defined",
        },
        overlap_threshold=0.5,
        existing_specs=[],
        max_attempts=2,
    )

    assert len(seen_payloads) == 2
    assert seen_payloads[1]["repair_context"]["mode"] == "contract_mismatch"
    assert "exact public entry-point" in seen_payloads[1]["repair_context"]["reason"]


def test_generate_with_regeneration_includes_missing_contracts_in_retry_payload() -> None:
    from codemint.synthesize.feasibility import FeasibilityResult
    from codemint.synthesize.pipeline import _generate_with_regeneration

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["function_name_mismatch"],
        frequency=1,
        sample_task_ids=[201],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Function name mismatches break execution.",
            capability_cliff="Strict harnesses require the exact public entry point.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.94,
        ),
    )
    seen_payloads: list[dict] = []

    def invoke_model(payload: dict) -> dict:
        seen_payloads.append(payload)
        return {
            "algorithm_type": "simulation",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": "The trap fails when `solve_value` is exposed instead of the required `solve` entry point.",
            "must_cover": ["exact callable entry point solve(x)"],
            "must_avoid": ["alternate public function names"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Implement the exact solve(x) entry point.",
                "common_wrong_approach": "Expose solve_value(x) instead of solve(x).",
                "distinguishing_test": "Call solve() directly from the harness.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        }

    feasibility_calls = {"count": 0}

    def feasibility_check(payload: dict) -> FeasibilityResult:
        feasibility_calls["count"] += 1
        if feasibility_calls["count"] == 1:
            return FeasibilityResult(
                accepted=False,
                reason="Function-name mismatch weakness spec must require a single exact public entry-point contract and forbid alternate public function names.",
                missing_contracts=[
                    "requires_exact_public_entry_point",
                    "forbids_alternate_public_names",
                ],
            )
        return FeasibilityResult(accepted=True, reason="ok")

    _generate_with_regeneration(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        spec_index=1,
        difficulty="medium",
        invoke_model=invoke_model,
        feasibility_check=feasibility_check,
        original_evidence={
            "wrong_line": "def solve_value(x):",
            "correct_approach": "Define the exact solve(x) entry point expected by the harness.",
            "failed_test": "NameError: name 'solve' is not defined",
        },
        overlap_threshold=0.5,
        existing_specs=[],
        max_attempts=2,
    )

    assert len(seen_payloads) == 2
    assert seen_payloads[1]["repair_context"]["mode"] == "contract_mismatch"
    assert seen_payloads[1]["repair_context"]["missing_contracts"] == [
        "requires_exact_public_entry_point",
        "forbids_alternate_public_names",
    ]


def test_generate_with_regeneration_includes_structured_repair_mode_for_raw_output_and_diversity() -> None:
    from codemint.synthesize.feasibility import FeasibilityResult
    from codemint.synthesize.pipeline import _generate_with_regeneration

    weakness = WeaknessEntry(
        rank=1,
        fault_type="surface",
        sub_tags=["markdown_formatting"],
        frequency=1,
        sample_task_ids=[202],
        trainability=0.3,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Markdown fences wrap otherwise executable code.",
            capability_cliff="Raw-output consumers fail when fence markers are present.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.9,
        ),
    )
    seen_payloads: list[dict] = []

    def invoke_model(payload: dict) -> dict:
        seen_payloads.append(payload)
        return {
            "algorithm_type": "simulation",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": "The trap reproduces the original ```python fenced solve(x) output instead of raw executable code.",
            "must_cover": ["raw executable code output"],
            "must_avoid": ["fenced code blocks"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Return raw executable code only.",
                "common_wrong_approach": "Wrap the answer in markdown fences.",
                "distinguishing_test": "Reject answers containing ```.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        }

    diversity_calls = {"count": 0}

    def feasibility_check(payload: dict) -> FeasibilityResult:
        return FeasibilityResult(accepted=True, reason="ok")

    from codemint.synthesize import pipeline as synth_pipeline

    original_assign = synth_pipeline.assign_diversity_tags

    def fake_assign(existing_specs, diversity_tags, overlap_threshold):
        diversity_calls["count"] += 1
        if diversity_calls["count"] == 1:
            return type("Result", (), {"accepted": False, "reason": "too similar to previous warehouse/array pattern"})()
        return original_assign(existing_specs, diversity_tags, overlap_threshold)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(synth_pipeline, "assign_diversity_tags", fake_assign)
    try:
        _generate_with_regeneration(
            weakness,
            diversity_tags=DiversityTags(
                narrative_theme="warehouses",
                data_structure="array",
                constraint_scale="small",
            ),
            spec_index=1,
            difficulty="medium",
            invoke_model=invoke_model,
            feasibility_check=feasibility_check,
            original_evidence={
                "wrong_line": "```python\ndef solve(x):\n    return x + 1\n```",
                "correct_approach": "Return raw executable code without markdown fences.",
                "failed_test": "SyntaxError: invalid syntax",
            },
            overlap_threshold=0.5,
            existing_specs=[],
            max_attempts=2,
        )
    finally:
        monkeypatch.undo()

    assert len(seen_payloads) == 2
    assert seen_payloads[1]["repair_context"]["mode"] == "duplicate_diversity_pattern"
    assert "too similar" in seen_payloads[1]["repair_context"]["reason"]


def test_generate_with_regeneration_includes_structured_repair_mode_for_syntax_error() -> None:
    from codemint.synthesize.feasibility import FeasibilityResult
    from codemint.synthesize.pipeline import _generate_with_regeneration

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["syntax_error"],
        frequency=1,
        sample_task_ids=[102],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="The generated code is syntactically invalid.",
            capability_cliff="Execution fails before semantic correctness can be tested.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.91,
        ),
    )
    seen_payloads: list[dict] = []

    def invoke_model(payload: dict) -> dict:
        seen_payloads.append(payload)
        return {
            "algorithm_type": "simulation",
            "difficulty": "medium",
            "narrative_theme": "warehouses",
            "constraints": {
                "n_range": [1, 100],
                "value_range": [0, 1000],
                "time_limit": "1s",
                "memory_limit": "256MB",
            },
            "key_trap": "The trap reproduces the original def solve(x) header without the required colon.",
            "must_cover": ["basic implementation correctness"],
            "must_avoid": ["verbatim reuse of prior tasks"],
            "verification_spec": {
                "min_test_cases": 4,
                "must_include_edge_cases": ["single value input"],
                "brute_force_verifiable": True,
                "brute_force_complexity_limit": "O(n^2)",
            },
            "generation_hints": {
                "solution_approach": "Return syntactically valid executable code.",
                "common_wrong_approach": "Omit the colon after the function header.",
                "distinguishing_test": "Parse the final code before execution.",
            },
            "language_constraint": {
                "target_languages": ["python"],
                "language_specific": False,
            },
        }

    feasibility_calls = {"count": 0}

    def feasibility_check(payload: dict) -> FeasibilityResult:
        feasibility_calls["count"] += 1
        if feasibility_calls["count"] == 1:
            return FeasibilityResult(
                accepted=False,
                reason="Syntax-error weakness spec must require syntactically complete executable code and forbid incomplete code forms.",
            )
        return FeasibilityResult(accepted=True, reason="ok")

    _generate_with_regeneration(
        weakness,
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        spec_index=1,
        difficulty="medium",
        invoke_model=invoke_model,
        feasibility_check=feasibility_check,
        original_evidence={
            "wrong_line": "def solve(x)",
            "correct_approach": "Add a colon at the end of the function definition: def solve(x):",
            "failed_test": "SyntaxError: expected ':'",
        },
        overlap_threshold=0.5,
        existing_specs=[],
        max_attempts=2,
    )

    assert len(seen_payloads) == 2
    assert seen_payloads[1]["repair_context"]["mode"] == "syntax_completion_required"
    assert "syntactically complete executable code" in seen_payloads[1]["repair_context"]["reason"]


def test_generate_or_log_failure_uses_builtin_fallback_for_markdown_formatting(tmp_path: Path) -> None:
    from codemint.synthesize.pipeline import _generate_or_log_failure

    weakness = WeaknessEntry(
        rank=1,
        fault_type="surface",
        sub_tags=["markdown_formatting"],
        frequency=1,
        sample_task_ids=[104],
        trainability=0.3,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="Markdown fences pollute raw code output.",
            capability_cliff="Execution fails when formatting wrappers are preserved.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.9,
        ),
    )

    spec = _generate_or_log_failure(
        weakness,
        output_path=tmp_path / "specs.jsonl",
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        spec_index=1,
        difficulty="medium",
        invoke_model=lambda payload: (_ for _ in ()).throw(ValueError("model failed")),
        feasibility_check=lambda payload: {"accepted": True, "reason": "ok"},
        original_evidence={
            "wrong_line": "```python\ndef solve(x):\n    return x - 1\n```",
            "correct_approach": "def solve(x):\n    return x - 1",
            "failed_test": "Execution environment failed to parse markdown syntax as Python.",
        },
        overlap_threshold=0.5,
        existing_specs=[],
        max_attempts=1,
    )

    assert spec is not None
    assert spec.target_weakness.sub_tags == ["markdown_formatting"]
    assert any("raw executable code" in item.lower() for item in spec.problem_spec.must_cover)


def test_generate_or_log_failure_uses_builtin_fallback_for_missing_code_block(tmp_path: Path) -> None:
    from codemint.synthesize.pipeline import _generate_or_log_failure

    weakness = WeaknessEntry(
        rank=1,
        fault_type="surface",
        sub_tags=["missing_code_block"],
        frequency=1,
        sample_task_ids=[101],
        trainability=0.3,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="The model emits explanation instead of executable code.",
            capability_cliff="Breaks when the task requires direct code output.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.93,
        ),
    )

    spec = _generate_or_log_failure(
        weakness,
        output_path=tmp_path / "specs.jsonl",
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        spec_index=1,
        difficulty="medium",
        invoke_model=lambda payload: (_ for _ in ()).throw(ValueError("model failed")),
        feasibility_check=lambda payload: {"accepted": True, "reason": "ok"},
        original_evidence={
            "wrong_line": "I would solve this by defining a function and returning x plus one, but the final code block is missing.",
            "correct_approach": "def solve(x):\n    return x + 1",
            "failed_test": "assert solve(1) == 2",
        },
        overlap_threshold=0.5,
        existing_specs=[],
        max_attempts=1,
    )

    assert spec is not None
    assert spec.target_weakness.sub_tags == ["missing_code_block"]
    assert any("executable code" in item.lower() for item in spec.problem_spec.must_cover)


def test_generate_or_log_failure_uses_builtin_fallback_for_syntax_error(tmp_path: Path) -> None:
    from codemint.synthesize.pipeline import _generate_or_log_failure

    weakness = WeaknessEntry(
        rank=1,
        fault_type="implementation",
        sub_tags=["syntax_error"],
        frequency=1,
        sample_task_ids=[102],
        trainability=0.6,
        collective_diagnosis=CollectiveDiagnosis(
            refined_root_cause="The generated code is syntactically invalid.",
            capability_cliff="Execution fails before semantic correctness can be tested.",
            misdiagnosed_ids=[],
            misdiagnosis_corrections={},
            cluster_coherence=0.91,
        ),
    )

    spec = _generate_or_log_failure(
        weakness,
        output_path=tmp_path / "specs.jsonl",
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="small",
        ),
        spec_index=1,
        difficulty="medium",
        invoke_model=lambda payload: (_ for _ in ()).throw(ValueError("model failed")),
        feasibility_check=lambda payload: {"accepted": True, "reason": "ok"},
        original_evidence={
            "wrong_line": "def solve(x)",
            "correct_approach": "Add a colon at the end of the function definition: def solve(x):",
            "failed_test": "SyntaxError: expected ':'",
        },
        overlap_threshold=0.5,
        existing_specs=[],
        max_attempts=1,
    )

    assert spec is not None
    assert spec.target_weakness.sub_tags == ["syntax_error"]
    assert any("syntactically complete executable code" in item.lower() for item in spec.problem_spec.must_cover)


def test_synthesize_emits_fine_grained_progress_per_slot(tmp_path: Path) -> None:
    from codemint.synthesize.pipeline import run_synthesize

    output_path = tmp_path / "specs.jsonl"
    events: list[dict[str, object]] = []

    specs = run_synthesize(
        _report(),
        output_path,
        config=CodeMintConfig.model_validate(
            {
                "synthesize": {
                    "specs_per_weakness": 2,
                    "max_per_weakness": 2,
                    "top_n": 1,
                    "narrative_themes": {"generic": ["warehouses", "ports"], "domain_adaptive": False},
                    "data_structures": ["array"],
                }
            }
        ),
        original_evidence={
            "state_tracking": {
                "wrong_line": "Dropped the carry before processing the final warehouse.",
                "correct_approach": "Preserve carry state through the final transition.",
                "failed_test": "Final warehouse requires carrying state into the last step.",
            }
        },
        progress_callback=events.append,
    )

    assert len(specs) == 2
    assert [event["processed"] for event in events] == [1, 2]
    assert all(event["total"] == 2 for event in events)


def test_synthesize_uses_model_client_when_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from codemint.synthesize.pipeline import run_synthesize

    output_path = tmp_path / "specs.jsonl"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return """{
              "algorithm_type": "dynamic programming",
              "difficulty": "hard",
              "narrative_theme": "warehouses",
              "constraints": {"n_range": [1, 100], "value_range": [0, 1000], "time_limit": "1s", "memory_limit": "256MB"},
              "key_trap": "Reference the original evidence: Dropped the carry before processing the final warehouse.",
              "must_cover": ["state_tracking", "Preserve carry state through the final transition."],
              "must_avoid": ["verbatim reuse of prior tasks"],
              "verification_spec": {"min_test_cases": 4, "must_include_edge_cases": ["Final warehouse requires carrying state into the last step."], "brute_force_verifiable": true, "brute_force_complexity_limit": "O(n^2)"},
              "generation_hints": {"solution_approach": "Preserve carry state through the final transition.", "common_wrong_approach": "Dropped the carry before processing the final warehouse.", "distinguishing_test": "Final warehouse requires carrying state into the last step."},
              "language_constraint": {"target_languages": ["python"], "language_specific": false}
            }"""

    monkeypatch.setattr("codemint.synthesize.pipeline.ModelClient", StubClient)

    specs = run_synthesize(
        _report(),
        output_path,
        config=CodeMintConfig.model_validate(
            {
                "model": {
                    "base_url": "https://example.test",
                    "api_key": "secret",
                    "analysis_model": "gpt-test",
                },
                "synthesize": {
                    "specs_per_weakness": 1,
                    "max_per_weakness": 1,
                    "top_n": 1,
                    "narrative_themes": {"generic": ["warehouses"], "domain_adaptive": False},
                    "data_structures": ["array"],
                },
            }
        ),
        original_evidence={
            "state_tracking": {
                "wrong_line": "Dropped the carry before processing the final warehouse.",
                "correct_approach": "Preserve carry state through the final transition.",
                "failed_test": "Final warehouse requires carrying state into the last step.",
            }
        },
    )

    assert len(specs) == 1
    assert specs[0].problem_spec.algorithm_type == "dynamic programming"


def test_synthesize_normalizes_real_model_style_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from codemint.synthesize.pipeline import run_synthesize

    output_path = tmp_path / "specs.jsonl"

    class StubClient:
        def __init__(self, config):
            self.config = config

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return """```json
            {
              "algorithm_type": "dynamic programming",
              "difficulty": "hard",
              "narrative_theme": "warehouses",
              "constraints": "Array size between 2 and 100. Values between 0 and 1000. Time limit 1s. Memory limit 256MB.",
              "key_trap": "Reference the original evidence: Dropped the carry before processing the final warehouse and preserve carry state through the final transition.",
              "must_cover": "state_tracking; Preserve carry state through the final transition.",
              "must_avoid": "verbatim reuse of prior tasks; sorting",
              "verification_spec": "At least 4 tests. Include edge case: Final warehouse requires carrying state into the last step. Brute force verifiable yes. Complexity O(n^2).",
              "generation_hints": "Solution approach: Preserve carry state through the final transition. Common wrong approach: Dropped the carry before processing the final warehouse. Distinguishing test: Final warehouse requires carrying state into the last step.",
              "language_constraint": "Python",
              "tasks": [{"task_id": "ignore-me"}]
            }
            ```"""

    monkeypatch.setattr("codemint.synthesize.pipeline.ModelClient", StubClient)

    specs = run_synthesize(
        _report(),
        output_path,
        config=CodeMintConfig.model_validate(
            {
                "model": {
                    "base_url": "https://example.test",
                    "api_key": "secret",
                    "analysis_model": "gpt-test",
                },
                "synthesize": {
                    "specs_per_weakness": 1,
                    "max_per_weakness": 1,
                    "top_n": 1,
                    "narrative_themes": {"generic": ["warehouses"], "domain_adaptive": False},
                    "data_structures": ["array"],
                },
            }
        ),
        original_evidence={
            "state_tracking": {
                "wrong_line": "Dropped the carry before processing the final warehouse.",
                "correct_approach": "Preserve carry state through the final transition.",
                "failed_test": "Final warehouse requires carrying state into the last step.",
            }
        },
    )

    assert len(specs) == 1
    assert specs[0].problem_spec.constraints.time_limit == "1s"
    assert specs[0].verification_spec.min_test_cases == 4
    assert specs[0].language_constraint.target_languages == ["python"]


def _report() -> WeaknessReport:
    return WeaknessReport(
        weaknesses=[
            WeaknessEntry(
                rank=1,
                fault_type="modeling",
                sub_tags=["state_tracking"],
                frequency=5,
                sample_task_ids=[1, 2, 3],
                trainability=0.9,
                collective_diagnosis=CollectiveDiagnosis(
                    refined_root_cause="State invariants break after transitions.",
                    capability_cliff="Long chains require carrying latent state.",
                    misdiagnosed_ids=[],
                    misdiagnosis_corrections={},
                    cluster_coherence=0.94,
                ),
            )
        ],
        rankings=RankingSet(by_frequency=[1], by_difficulty=[1], by_trainability=[1]),
        causal_chains=[CausalChain(root="state_tracking", downstream=[], training_priority="high")],
        tag_mappings={"carry_state": "state_tracking", "state_tracking": "state_tracking"},
    )


def _diagnosis() -> DiagnosisRecord:
    return DiagnosisRecord(
        task_id=11,
        fault_type="modeling",
        sub_tags=["carry_state"],
        severity="high",
        description="State is lost at the final transition.",
        evidence=DiagnosisEvidence(
            wrong_line="Dropped the carry before processing the final warehouse.",
            correct_approach="Preserve carry state through the final transition.",
            failed_test="Final warehouse requires carrying state into the last step.",
        ),
        enriched_labels={},
        confidence=0.9,
        diagnosis_source="model_deep",
        prompt_version="test-v1",
    )


def _existing_spec() -> SpecRecord:
    return SpecRecord(
        spec_id="existing-0001",
        target_weakness=TargetWeakness(
            fault_type="modeling",
            sub_tags=["state_tracking"],
            root_cause="Existing root cause",
            capability_cliff="Existing cliff",
        ),
        problem_spec=ProblemSpec(
            algorithm_type="dynamic programming",
            difficulty="hard",
            narrative_theme="warehouses",
            constraints=ProblemConstraints(
                n_range=[1, 100],
                value_range=[0, 1000],
                time_limit="1s",
                memory_limit="256MB",
            ),
            key_trap="Reference `carry` before final warehouse processing.",
            must_cover=["state_tracking"],
            must_avoid=["duplicate"],
        ),
        verification_spec=VerificationSpec(
            min_test_cases=4,
            must_include_edge_cases=["single warehouse"],
            brute_force_verifiable=True,
            brute_force_complexity_limit="O(n^2)",
        ),
        diversity_tags=DiversityTags(
            narrative_theme="warehouses",
            data_structure="array",
            constraint_scale="medium",
        ),
        generation_hints=GenerationHints(
            solution_approach="Track carry state",
            common_wrong_approach="Drop state early",
            distinguishing_test="Final warehouse needs carry",
        ),
        language_constraint=LanguageConstraint(
            target_languages=["python"],
            language_specific=False,
        ),
        prompt_version="v1",
    )
