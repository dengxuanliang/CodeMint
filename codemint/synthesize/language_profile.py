from __future__ import annotations

import re

from codemint.models.base import StrictModel


SUPPORTED_LANGUAGES = {
    "python",
    "r",
    "java",
    "javascript",
    "typescript",
    "cpp",
    "go",
    "rust",
    "unknown",
}

FENCE_LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "r": "r",
    "java": "java",
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "cpp": "cpp",
    "c++": "cpp",
    "go": "go",
    "rust": "rust",
    "rs": "rust",
}

SYNTAX_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "python",
        (
            re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE),
            re.compile(r"^\s*import\s+[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*\s*$", re.MULTILINE),
        ),
    ),
    (
        "r",
        (
            re.compile(r"<-\s*function\s*\("),
            re.compile(r"\bfunction\s*\([^)]*\)\s*[^{\n]*$"),
        ),
    ),
    (
        "java",
        (
            re.compile(r"\bpublic\s+static\s+\w[\w<>\[\]]*\s+\w+\s*\("),
            re.compile(r"\bSystem\.out\.println\s*\("),
        ),
    ),
    (
        "typescript",
        (
            re.compile(r"\bfunction\s+\w+\s*\([^)]*:\s*[A-Za-z_][\w<>\[\]\|]*"),
            re.compile(r"\b(?:const|let)\s+\w+\s*:\s*[A-Za-z_][\w<>\[\]\|]*\s*="),
            re.compile(r"\binterface\s+\w+\b"),
        ),
    ),
    (
        "javascript",
        (
            re.compile(r"^\s*import\s+.+\s+from\s+['\"][^'\"]+['\"];?\s*$", re.MULTILINE),
            re.compile(r"\bfunction\s+\w+\s*\("),
            re.compile(r"\bconsole\.log\s*\("),
            re.compile(r"\b(?:const|let)\s+\w+\s*=\s*\("),
        ),
    ),
    (
        "cpp",
        (
            re.compile(r"^\s*#include\s*<[^>]+>", re.MULTILINE),
            re.compile(r"\bstd::\w+"),
        ),
    ),
    (
        "go",
        (
            re.compile(r"^\s*func\s+\w+\s*\(", re.MULTILINE),
            re.compile(r"^\s*package\s+[A-Za-z_]\w*\s*$", re.MULTILINE),
        ),
    ),
    (
        "rust",
        (
            re.compile(r"^\s*fn\s+\w+\s*\(", re.MULTILINE),
            re.compile(r"\blet\s+mut\s+\w+"),
        ),
    ),
)

EXPLICIT_LANGUAGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (
        language,
        re.compile(rf"\b{re.escape(language)}\b", re.IGNORECASE),
    )
    for language in ("python", "r", "java", "javascript", "typescript", "go", "rust")
) + (
    ("cpp", re.compile(r"\bc\+\+\b|\bcpp\b", re.IGNORECASE)),
)


class LanguageProfile(StrictModel):
    primary_language: str
    target_languages: list[str]
    language_specific: bool


def infer_language_profile(representative_evidence: dict[str, str]) -> LanguageProfile:
    text = _flatten_evidence(representative_evidence)
    language = (
        _infer_from_fences(text)
        or _infer_from_syntax(text)
        or _infer_from_explicit_mentions(text)
        or "unknown"
    )
    return LanguageProfile(
        primary_language=language,
        target_languages=[language],
        language_specific=language in SUPPORTED_LANGUAGES and language != "unknown",
    )


def _flatten_evidence(representative_evidence: dict[str, str]) -> str:
    return "\n".join(
        representative_evidence.get(field, "")
        for field in ("wrong_line", "correct_approach", "failed_test")
    )


def _infer_from_fences(text: str) -> str | None:
    for match in re.finditer(r"```([A-Za-z0-9#+-]+)", text):
        alias = match.group(1).strip().lower()
        language = FENCE_LANGUAGE_ALIASES.get(alias)
        if language is not None:
            return language
    return None


def _infer_from_syntax(text: str) -> str | None:
    for language, patterns in SYNTAX_PATTERNS:
        if any(pattern.search(text) for pattern in patterns):
            return language
    return None


def _infer_from_explicit_mentions(text: str) -> str | None:
    for language, pattern in EXPLICIT_LANGUAGE_PATTERNS:
        if pattern.search(text):
            return language
    return None
