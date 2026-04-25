from __future__ import annotations

from codemint.models.base import StrictModel
from codemint.models.weakness import WeaknessEntry
from codemint.synthesize.allocation import weakness_key
from codemint.synthesize.language_profile import infer_language_profile


FIXED_SUMMARIES = {
    "missing_code_block": "Model outputs explanation or transformed prompt instead of executable code.",
    "function_name_mismatch": "Model violates the required public entrypoint contract.",
    "markdown_formatting": "Model wraps otherwise executable code in markdown fences.",
    "syntax_error": "Model emits code that is not syntactically executable.",
    "non_executable_code": "Model emits code-like output that still cannot run directly.",
}


class SynthesisInputView(StrictModel):
    fault_type: str
    primary_sub_tag: str
    frequency: int
    sample_task_ids: list[int]
    canonical_summary: str
    representative_evidence: dict[str, str]
    primary_language: str
    target_languages: list[str]
    language_specific: bool


def build_synthesis_input_view(
    weakness: WeaknessEntry,
    representative_evidence: dict[str, str],
) -> SynthesisInputView:
    primary_sub_tag = weakness_key(weakness)
    language_profile = infer_language_profile(representative_evidence)
    return SynthesisInputView(
        fault_type=weakness.fault_type,
        primary_sub_tag=primary_sub_tag,
        frequency=weakness.frequency,
        sample_task_ids=list(weakness.sample_task_ids),
        canonical_summary=canonical_summary_for_weakness(primary_sub_tag, representative_evidence),
        representative_evidence=representative_evidence,
        primary_language=language_profile.primary_language,
        target_languages=language_profile.target_languages,
        language_specific=language_profile.language_specific,
    )


def canonical_summary_for_weakness(
    primary_sub_tag: str,
    representative_evidence: dict[str, str],
) -> str:
    if primary_sub_tag in FIXED_SUMMARIES:
        return FIXED_SUMMARIES[primary_sub_tag]
    if primary_sub_tag == "logic_error":
        return _logic_error_summary(representative_evidence)
    return "Model computes the wrong result with executable logic."


def _logic_error_summary(representative_evidence: dict[str, str]) -> str:
    text = " ".join(
        representative_evidence.get(field, "")
        for field in ("wrong_line", "correct_approach", "failed_test")
    ).lower()

    if any(
        token in text
        for token in ("return only", "required sorted list", "expected contract", "required format")
    ):
        return "Model returns an output that violates the expected contract."
    if any(token in text for token in ("edge", "boundary", "n-1", "empty", "single element", "terminal index")):
        return "Model fails on a required boundary or edge condition."
    if any(token in text for token in ("flat list", "nested", "dictionary", "dict", "tuple", "wrong structure", "structure")):
        return "Model returns the wrong structure despite executable code."
    if any(
        token in text
        for token in ("keyerror", "typeerror", "missing key", "missing keys", "mapping", "map", "mutate", "schema")
    ):
        return "Model misuses an API or data contract in executable code."
    return "Model computes the wrong result with executable logic."
