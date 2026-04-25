from __future__ import annotations

from pydantic import Field

from codemint.models.base import StrictModel
from codemint.models.spec import SpecRecord


class ContractSignals(StrictModel):
    requires_exact_public_entry_point: bool = False
    forbids_alternate_public_names: bool = False
    requires_executable_code_output: bool = False
    forbids_explanation_only: bool = False
    requires_raw_executable_output: bool = False
    forbids_markdown_wrapping: bool = False
    requires_syntactic_completeness: bool = False
    forbids_incomplete_code: bool = False
    matched_phrases: list[str] = Field(default_factory=list)


def extract_contract_signals(spec: SpecRecord) -> ContractSignals:
    cover_items = spec.problem_spec.must_cover
    avoid_items = spec.problem_spec.must_avoid
    cover_text = " ".join(cover_items).lower()
    avoid_text = " ".join(avoid_items).lower()
    matched_phrases: list[str] = []

    return ContractSignals(
        requires_exact_public_entry_point=_match_any(
            cover_text,
            matched_phrases,
            "exact callable entry point",
            "exact public entry point",
            "exact public function named",
            "exact public method",
            "exact public attribute",
            "exact public symbol",
            "exactly one public callable named",
            "exactly one public callable entry point",
            "single public function named solve",
            "public function named solve",
            "exact solver entrypoint",
            "exact entry point",
            "harness entry point",
            "solve(x)",
        ),
        forbids_alternate_public_names=_match_any(
            avoid_text,
            matched_phrases,
            "alternate public function names",
            "alternate public method names",
            "alternate public attribute names",
            "alternate public symbol names",
            "alternate public names",
            "do not expose alternate public function names",
            "do not rename",
            "rename the public callable",
            "helper entrypoint aliases",
            "solve_value",
            "solver",
        ),
        requires_executable_code_output=_match_any(
            cover_text,
            matched_phrases,
            "executable code",
            "executable java code",
            "executable javascript code",
            "executable typescript code",
            "executable c++ code",
            "executable go code",
            "executable rust code",
            "executable r code",
            "runnable code",
            "callable solution entry point",
            "runnable solve function",
            "callable solve function",
            "implementation output",
            "emitted executable implementation output",
        ),
        forbids_explanation_only=_match_any(
            avoid_text,
            matched_phrases,
            "explanation-only",
            "prose",
            "natural language only",
            "instead of executable code",
            "do not return explanation-only",
        ),
        requires_raw_executable_output=_match_any(
            cover_text,
            matched_phrases,
            "raw executable code",
            "raw code output",
            "plain executable program text",
            "plain executable code",
            "raw output",
            "raw solve",
        ),
        forbids_markdown_wrapping=_match_any(
            avoid_text,
            matched_phrases,
            "markdown code fences",
            "markdown fences",
            "backticks",
            "formatting delimiters",
            "wrapping delimiters",
            "code fences",
        ),
        requires_syntactic_completeness=_match_any(
            cover_text,
            matched_phrases,
            "syntactically complete",
            "syntactically valid",
            "parseable",
            "valid solve(x) definition",
            "function definition must be complete",
        ),
        forbids_incomplete_code=_match_any(
            avoid_text,
            matched_phrases,
            "incomplete code",
            "missing colons",
            "omit required punctuation",
            "malformed",
            "partial function headers",
            "missing bodies",
        ),
        matched_phrases=matched_phrases,
    )


def _match_any(text: str, matched_phrases: list[str], *phrases: str) -> bool:
    for phrase in phrases:
        if phrase in text:
            matched_phrases.append(phrase)
            return True
    return False
