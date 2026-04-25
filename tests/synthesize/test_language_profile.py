from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec


def test_supported_languages_matches_bounded_plan() -> None:
    module = _language_profile_module()

    assert module.SUPPORTED_LANGUAGES == {
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


def test_infer_language_from_python_fence() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "```python\ndef solve(x):\n    return x + 1\n```",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "python"
    assert profile.target_languages == ["python"]
    assert profile.language_specific is True


def test_infer_language_from_python_fence_alias() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "```py\ndef solve(x):\n    return x + 1\n```",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "python"
    assert profile.target_languages == ["python"]
    assert profile.language_specific is True


def test_infer_language_from_r_fence() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "```r\nsolve <- function(x) x + 1\n```",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "r"
    assert profile.target_languages == ["r"]
    assert profile.language_specific is True


def test_infer_language_from_java_syntax() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "public static int solve(int x) { return x + 1; }",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "java"
    assert profile.target_languages == ["java"]
    assert profile.language_specific is True


def test_infer_language_from_javascript_syntax() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "function solve(x) { return x + 1; }\nconsole.log(solve(1));",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "javascript"
    assert profile.target_languages == ["javascript"]
    assert profile.language_specific is True


def test_es_module_import_is_not_misclassified_as_python() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "import { solve } from './solver';",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "javascript"
    assert profile.target_languages == ["javascript"]
    assert profile.language_specific is True


def test_infer_language_from_typescript_syntax() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "function solve(x: number): number { return x + 1; }",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "typescript"
    assert profile.target_languages == ["typescript"]
    assert profile.language_specific is True


def test_infer_language_from_cpp_syntax() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "#include <vector>\nint solve(int x) { return x + 1; }",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "cpp"
    assert profile.target_languages == ["cpp"]
    assert profile.language_specific is True


def test_infer_language_from_go_syntax() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "func solve(x int) int {\n\treturn x + 1\n}",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "go"
    assert profile.target_languages == ["go"]
    assert profile.language_specific is True


def test_java_package_declaration_is_not_misclassified_as_go() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "package com.example.tools;\nclass Solver {}",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "unknown"
    assert profile.target_languages == ["unknown"]
    assert profile.language_specific is False


def test_infer_language_from_rust_syntax() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "fn solve(x: i32) -> i32 { x + 1 }",
            "correct_approach": "",
            "failed_test": "",
        }
    )

    assert profile.primary_language == "rust"
    assert profile.target_languages == ["rust"]
    assert profile.language_specific is True


def test_unknown_evidence_returns_unknown_language() -> None:
    profile = _infer_language_profile(
        {
            "wrong_line": "Sort the numbers before returning them.",
            "correct_approach": "Use the required algorithm.",
            "failed_test": "Wrong answer on sample 2.",
        }
    )

    assert profile.primary_language == "unknown"
    assert profile.target_languages == ["unknown"]
    assert profile.language_specific is False


def _infer_language_profile(representative_evidence: dict[str, str]):
    module = _language_profile_module()
    return module.infer_language_profile(representative_evidence)


def _language_profile_module():
    module_spec = find_spec("codemint.synthesize.language_profile")
    assert module_spec is not None
    return import_module("codemint.synthesize.language_profile")
