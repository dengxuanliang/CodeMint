from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from codemint.models.diagnosis import FaultType, Severity


@dataclass(frozen=True)
class DiagnosisRule:
    rule_id: str
    pattern: Pattern[str]
    fault_type: FaultType
    sub_tag: str
    severity: Severity
    priority: int
    requires_model_analysis: bool = False


def _rule(
    rule_id: str,
    pattern: str,
    fault_type: FaultType,
    sub_tag: str,
    severity: Severity,
    priority: int,
    requires_model_analysis: bool = False,
) -> DiagnosisRule:
    return DiagnosisRule(
        rule_id=rule_id,
        pattern=re.compile(pattern, re.IGNORECASE | re.MULTILINE),
        fault_type=fault_type,
        sub_tag=sub_tag,
        severity=severity,
        priority=priority,
        requires_model_analysis=requires_model_analysis,
    )


def default_rules() -> list[DiagnosisRule]:
    return [
        _rule(
            "R007",
            r"\b(timeout|time\s*limit\s*exceeded|tle|timed\s*out|TimeoutError)\b",
            "modeling",
            "time_limit_exceeded",
            "high",
            1,
        ),
        _rule(
            "R006",
            r"\b(RecursionError|StackOverflowError|stack\s*overflow|maximum\s+recursion\s+depth)\b",
            "modeling",
            "recursion_depth_exceeded",
            "high",
            2,
        ),
        _rule(
            "R009",
            r"(\berror:\s|cannot\s+find\s+symbol|expected\s+['\";}]|undefined\s+reference|"
            r"compilation\s+(error|failed)|compile\s+(error|failed)|"
            r"javac|g\+\+|gcc|go:\s|syntax\s+error:\s+unexpected)",
            "surface",
            "compilation_error",
            "high",
            3,
        ),
        _rule(
            "R001",
            r"\b(SyntaxError|IndentationError|TabError|unexpected\s+indent|invalid\s+syntax)\b",
            "surface",
            "syntax_error",
            "high",
            4,
        ),
        _rule(
            "R002",
            r"\b(NameError|undefined\s+(name|variable)|not\s+defined|cannot\s+find\s+name)\b",
            "surface",
            "undefined_variable",
            "medium",
            5,
        ),
        _rule(
            "R003",
            r"\b(ImportError|ModuleNotFoundError|No\s+module\s+named|package\s+.*\s+does\s+not\s+exist)\b",
            "surface",
            "missing_import",
            "medium",
            6,
        ),
        _rule(
            "R004",
            r"\b(TypeError|argument\s+mismatch|wrong\s+number\s+of\s+arguments|"
            r"missing\s+\d+\s+required\s+positional\s+argument|takes\s+\d+\s+.*arguments?)\b",
            "implementation",
            "argument_mismatch",
            "medium",
            7,
        ),
        _rule(
            "R008",
            r"\b(empty\s+output|no\s+output|missing\s+return|NoneType|returned\s+None)\b",
            "implementation",
            "missing_return_or_empty_output",
            "medium",
            8,
        ),
        _rule(
            "R012",
            r"\b(ZeroDivisionError|division\s+by\s+zero|divide\s+by\s+zero|"
            r"ArithmeticException:\s*/\s+by\s+zero|math\s+domain\s+error)\b",
            "edge_handling",
            "invalid_math_edge_case",
            "medium",
            9,
        ),
        _rule(
            "R005",
            r"\b(IndexError|KeyError|ArrayIndexOutOfBoundsException|IndexOutOfBoundsException|"
            r"NoSuchElementException|index\s+out\s+of\s+range|key\s+not\s+found)\b",
            "edge_handling",
            "missing_bounds_or_key_check",
            "medium",
            10,
        ),
        _rule(
            "R011",
            r"\b(output\s+format\s+(mismatch|error)|presentation\s+error|wrong\s+format|"
            r"formatting\s+(issue|error))\b",
            "surface",
            "output_format_mismatch",
            "low",
            11,
        ),
        _rule(
            "R010",
            r"\b(AssertionError|assertion\s+failed|assert\s+failed)\b",
            "modeling",
            "assertion_failure",
            "medium",
            12,
            requires_model_analysis=True,
        ),
    ]
