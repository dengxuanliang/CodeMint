from __future__ import annotations

import re
from dataclasses import dataclass

from codemint.models.weakness import WeaknessEntry


@dataclass(frozen=True)
class PublicContractTarget:
    kind: str
    name: str
    owner: str | None = None


def normalize_contracts(
    weakness: WeaknessEntry,
    *,
    must_cover: list[str],
    must_avoid: list[str],
    target_languages: list[str] | None = None,
    context_texts: list[str] | None = None,
    original_evidence: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    key = weakness.sub_tags[0] if weakness.sub_tags else ""
    source_items = [
        *must_cover,
        *(context_texts or []),
        weakness.collective_diagnosis.refined_root_cause,
        weakness.collective_diagnosis.capability_cliff,
    ]
    entrypoint = _resolve_entrypoint(source_items)
    language = (target_languages or ["python"])[0]

    filtered_cover = _filter_process_constraints(must_cover)
    filtered_avoid = _filter_process_constraints(must_avoid)

    if key == "function_name_mismatch":
        target = _resolve_function_name_mismatch_target(
            source_items,
            original_evidence=original_evidence or {},
        )
        return _normalize_function_name_mismatch(target)
    if key == "missing_code_block":
        return _normalize_missing_code_block(entrypoint, language)
    if key == "markdown_formatting":
        return _normalize_markdown_formatting()
    if key == "syntax_error":
        return _normalize_syntax_error(entrypoint, language)
    if key == "logic_error":
        return _normalize_logic_error()
    if key == "non_executable_code":
        return _normalize_non_executable_code()

    return filtered_cover, filtered_avoid


def resolve_function_name_mismatch_target(
    *,
    must_cover: list[str],
    context_texts: list[str] | None = None,
    original_evidence: dict[str, str] | None = None,
) -> PublicContractTarget | None:
    return _resolve_function_name_mismatch_target(
        [*must_cover, *(context_texts or [])],
        original_evidence=original_evidence or {},
    )


def describe_public_contract_target(target: PublicContractTarget | None) -> str:
    if not target:
        return "the exact public entry point expected by the harness"
    if target.kind == "function":
        return f"the exact public function `{target.name}`"
    if target.kind == "method":
        return f"the exact public method `{target.name}`"
    if target.kind == "attribute":
        return f"the exact public attribute `{target.name}`"
    if target.kind == "symbol":
        return f"the exact public symbol `{target.name}`"
    return "the exact public entry point expected by the harness"


def describe_wrong_public_contract_target(original_evidence: dict[str, str]) -> str:
    wrong_line = original_evidence.get("wrong_line", "")
    wrong_dotted = _extract_expected_dotted_identifier_candidates(wrong_line)
    if wrong_dotted:
        return f"the wrong public attribute `{wrong_dotted[0]}`"
    wrong_entrypoints = _extract_entrypoint_candidates(wrong_line)
    if wrong_entrypoints:
        return f"the wrong public function `{wrong_entrypoints[0]}`"
    wrong_identifiers = _extract_expected_identifier_candidates(wrong_line)
    if wrong_identifiers:
        return f"the wrong public symbol `{wrong_identifiers[0]}`"
    wrong_fragment = wrong_line.strip()
    if wrong_fragment:
        return f"the wrong public output shape from `{wrong_fragment[:80]}`"
    return "the wrong public contract"


def _normalize_function_name_mismatch(target: PublicContractTarget | None) -> tuple[list[str], list[str]]:
    if target and target.kind == "function":
        return (
            [f"Require an exact public function named {target.name}."],
            ["Do not expose alternate public function names."],
        )
    if target and target.kind == "method":
        return (
            [f"Require the exact public method {target.name} on the expected object or solution class."],
            ["Do not expose alternate public method names."],
        )
    if target and target.kind == "attribute":
        return (
            [f"Require the exact public attribute {target.name} on the expected object."],
            ["Do not expose alternate public attribute names."],
        )
    if target and target.kind == "symbol":
        return (
            [f"Require the exact public symbol {target.name}."],
            ["Do not expose alternate public symbol names."],
        )
    return (
        ["Require the exact public entry point expected by the harness."],
        ["Do not expose alternate public names."],
    )


def _normalize_missing_code_block(
    entrypoint: str | None,
    language: str,
) -> tuple[list[str], list[str]]:
    cover = [f"Require {_language_code_phrase(language)} output."]
    if entrypoint:
        if language == "java":
            cover.append(f"Require exactly one callable solution method named {entrypoint}.")
        else:
            cover.append(f"Require exactly one callable solution entry point named {entrypoint}.")
    else:
        cover.append("Require exactly one callable solution entry point.")
    avoid = ["Do not return explanation-only or prose-only output."]
    return cover, avoid


def _normalize_markdown_formatting() -> tuple[list[str], list[str]]:
    return (
        ["Require raw executable code output."],
        ["Do not wrap the final answer in markdown fences, backticks, or fenced code blocks."],
    )


def _normalize_syntax_error(
    entrypoint: str | None,
    language: str,
) -> tuple[list[str], list[str]]:
    if language == "python":
        cover = ["Require syntactically complete executable code."]
    else:
        cover = [f"Require syntactically complete {_language_code_phrase(language)}."]
    if entrypoint:
        cover.append(f"Require a valid callable definition for {entrypoint}.")
    if language == "java":
        avoid = ["Do not emit incomplete or malformed code such as missing braces or malformed method headers."]
    elif language in {"cpp", "javascript", "typescript", "go", "rust", "r"}:
        avoid = ["Do not emit incomplete or malformed code such as missing required punctuation or malformed declarations."]
    else:
        avoid = ["Do not emit incomplete or malformed code such as missing colons, missing bodies, or malformed function headers."]
    return cover, avoid


def _normalize_logic_error() -> tuple[list[str], list[str]]:
    return (
        ["Require the exact output semantics specified by the task."],
        ["Do not transform the required result into a different structure, type, or semantic form."],
    )


def _normalize_non_executable_code() -> tuple[list[str], list[str]]:
    return (
        ["Require runnable code output."],
        ["Do not return non-executable prose, translation, or prompt echo instead of code."],
    )


def _filter_process_constraints(items: list[str]) -> list[str]:
    filtered: list[str] = []
    for item in items:
        normalized = item.strip()
        lowered = normalized.lower()
        if not normalized:
            continue
        if lowered.startswith("avoid duplicates of "):
            continue
        if lowered.startswith("repair feasibility issue:"):
            continue
        if lowered.startswith("repair diversity issue:"):
            continue
        filtered.append(normalized)
    return _dedupe(filtered)


def _resolve_entrypoint(items: list[str]) -> str | None:
    text = " ".join(items)
    for pattern in (
        r"`([A-Za-z_][A-Za-z0-9_]*)`\s+entry point",
        r"`([A-Za-z_][A-Za-z0-9_]*)`\s+method",
        r"\bexact callable entry point\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bentry point\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bpublic\s+static\s+[A-Za-z_][A-Za-z0-9_<>,\[\]\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    ):
        for match in re.findall(pattern, text):
            name = str(match)
            if name.lower() not in {
                "function",
                "entry",
                "point",
                "named",
                "definition",
                "callable",
                "public",
            }:
                return name
    return None


def _resolve_function_name_mismatch_target(
    items: list[str],
    *,
    original_evidence: dict[str, str],
) -> PublicContractTarget | None:
    for field in ("correct_approach", "failed_test"):
        target = _extract_public_contract_target(original_evidence.get(field, ""))
        if target:
            return target

    wrong_candidates: list[PublicContractTarget] = []
    expected_candidates: list[PublicContractTarget] = []
    neutral_candidates: list[PublicContractTarget] = []

    for item in items:
        text = item.strip()
        lowered = text.lower()
        if not text:
            continue
        natural_expected = _extract_public_contract_target(text)
        candidates = _extract_all_public_contract_targets(text)
        if natural_expected:
            expected_candidates.append(natural_expected)
        if not candidates:
            continue
        if any(marker in lowered for marker in ("instead of", "expected by the harness", "checker called", "calls only", "required by the harness")):
            expected_candidates.extend(candidates)
            wrong_fragment = text.split("instead of", 1)[0] if "instead of" in lowered else ""
            wrong_candidates.extend(_extract_all_public_contract_targets(wrong_fragment))
            continue
        if any(marker in lowered for marker in ("wrong callable", "expose compute", "model exposed", "nameerror")):
            wrong_candidates.extend(candidates)
            continue
        neutral_candidates.extend(candidates)

    for candidates in (expected_candidates, neutral_candidates):
        if candidates:
            return candidates[0]

    fallback_target = _extract_public_contract_target(original_evidence.get("wrong_line", ""))
    if fallback_target:
        return fallback_target

    if wrong_candidates:
        return wrong_candidates[0]
    return None


def _extract_public_contract_target(text: str) -> PublicContractTarget | None:
    targets = _extract_all_public_contract_targets(text)
    return targets[0] if targets else None


def _extract_all_public_contract_targets(text: str) -> list[PublicContractTarget]:
    targets: list[PublicContractTarget] = []

    for dotted in _extract_expected_dotted_identifier_candidates(text):
        owner, _, name = dotted.rpartition(".")
        if owner and name:
            _append_unique_target(targets, PublicContractTarget(kind="attribute", name=dotted, owner=f"the expected object `{owner}`"))

    for candidate in _extract_expected_identifier_candidates(text):
        _append_unique_target(targets, _classify_identifier_target(candidate, text))

    for candidate in _extract_entrypoint_candidates(text):
        _append_unique_target(targets, PublicContractTarget(kind="function", name=candidate))

    return targets


def _extract_entrypoint_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (
        r"`([A-Za-z_][A-Za-z0-9_]*)`\s+entry point",
        r"`([A-Za-z_][A-Za-z0-9_]*)`\s+method",
        r"\bexact callable entry point\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bentry point\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bpublic\s+static\s+[A-Za-z_][A-Za-z0-9_<>,\[\]\s]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bNameError:\s+name\s+['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    ):
        for match in re.findall(pattern, text):
            name = str(match)
            if name.lower() in {
                "function",
                "entry",
                "point",
                "named",
                "definition",
                "callable",
                "public",
            }:
                continue
            if name not in candidates:
                candidates.append(name)
    return candidates


def _extract_expected_identifier_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    patterns = (
        r"expected\s+the\s+exact\s+public\s+identifier\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
        r"expected\s+by\s+the\s+harness\s*[: ]+\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
        r"expected\s+callable\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
        r"imports?\s+only\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
        r"cannot import name\s+['`]([A-Za-z_][A-Za-z0-9_]*)['`]",
    )
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            name = str(match)
            if name not in candidates:
                candidates.append(name)
    return candidates


def _extract_expected_dotted_identifier_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    patterns = (
        r"expected\s+the\s+exact\s+public\s+identifier\s+`?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`?",
        r"exact\s+public\s+attribute\s+`?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`?",
        r"accesses?\s+only\s+`?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`?",
    )
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            name = str(match)
            if name not in candidates:
                candidates.append(name)
    return candidates


def _classify_identifier_target(candidate: str, text: str) -> PublicContractTarget:
    lowered = text.lower()
    if "." in candidate:
        owner, _, _ = candidate.rpartition(".")
        return PublicContractTarget(kind="attribute", name=candidate, owner=f"the expected object `{owner}`")
    if " method " in lowered or "public method" in lowered:
        return PublicContractTarget(kind="method", name=candidate)
    if any(marker in lowered for marker in ("callable", "function", "entry point", "entrypoint", "def ")):
        return PublicContractTarget(kind="function", name=candidate)
    return PublicContractTarget(kind="symbol", name=candidate)


def _append_unique_target(targets: list[PublicContractTarget], target: PublicContractTarget) -> None:
    if target not in targets:
        targets.append(target)


def _language_code_phrase(language: str) -> str:
    if language == "java":
        return "executable Java code"
    if language == "javascript":
        return "executable JavaScript code"
    if language == "typescript":
        return "executable TypeScript code"
    if language == "cpp":
        return "executable C++ code"
    if language == "go":
        return "executable Go code"
    if language == "rust":
        return "executable Rust code"
    if language == "r":
        return "executable R code"
    return "executable code"


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        lowered = item.strip().lower()
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(item)
    return deduped
