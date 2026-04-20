import re

from codemint.config import CustomPatternConfig, RulesConfig
from codemint.rules.builtin import DiagnosisRule
from codemint.rules.custom import build_rules
from codemint.rules.engine import RuleEngine


def test_priority_first_match_wins() -> None:
    result = RuleEngine(rules=build_rules()).match("TimeoutError and NameError")

    assert result is not None
    assert result.rule_id == "R007"


def test_engine_returns_none_when_no_rule_matches() -> None:
    result = RuleEngine(rules=build_rules()).match("accepted")

    assert result is None


def test_engine_matches_custom_rule() -> None:
    rules = build_rules(
        RulesConfig(
            custom_patterns=[
                CustomPatternConfig(
                    name="custom_memory_limit",
                    pattern="Memory Limit Exceeded|MLE",
                    fault_type="implementation",
                    sub_tag="memory_limit_exceeded",
                    severity="high",
                )
            ]
        )
    )

    result = RuleEngine(rules=rules).match("MLE on hidden test")

    assert result is not None
    assert result.rule_id == "custom_memory_limit"


def test_engine_respects_custom_priority_order() -> None:
    rules = build_rules(RulesConfig(rule_priority=["R010", "R007"]))

    result = RuleEngine(rules=rules).match("AssertionError after TimeoutError")

    assert result is not None
    assert result.rule_id == "R010"


def test_direct_engine_order_is_deterministic_for_duplicate_priorities() -> None:
    rules = [
        DiagnosisRule(
            rule_id="Z999",
            pattern=re.compile("same"),
            fault_type="surface",
            sub_tag="z_rule",
            severity="low",
            priority=1,
        ),
        DiagnosisRule(
            rule_id="A999",
            pattern=re.compile("same"),
            fault_type="surface",
            sub_tag="a_rule",
            severity="low",
            priority=1,
        ),
    ]

    result = RuleEngine(rules=rules).match("same")

    assert result is not None
    assert result.rule_id == "A999"
