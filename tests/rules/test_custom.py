from codemint.config import CustomPatternConfig, RulesConfig
from codemint.rules.custom import build_rules


def test_disabled_rule_is_removed_from_engine() -> None:
    rules = build_rules(RulesConfig(disabled_rules=["R003"]))

    assert "R003" not in [rule.rule_id for rule in rules]


def test_disabled_rule_can_be_passed_directly() -> None:
    rules = build_rules(disabled_rules=["R003"])

    assert "R003" not in [rule.rule_id for rule in rules]


def test_custom_patterns_are_merged_with_builtins() -> None:
    config = RulesConfig(
        custom_patterns=[
            CustomPatternConfig(
                name="custom_timeout",
                pattern="Time Limit Exceeded",
                fault_type="implementation",
                sub_tag="time_complexity_exceeded",
                severity="high",
            )
        ]
    )

    rules = build_rules(config)

    assert "custom_timeout" in [rule.rule_id for rule in rules]


def test_severity_override_applies_by_sub_tag() -> None:
    config = RulesConfig(severity_overrides={"missing_import": "low"})

    rules = build_rules(config)
    by_id = {rule.rule_id: rule for rule in rules}

    assert by_id["R003"].severity == "low"


def test_rule_priority_reorders_rules_without_losing_others() -> None:
    config = RulesConfig(rule_priority=["R010", "R007"])

    rules = build_rules(config)

    assert [rule.rule_id for rule in rules[:3]] == ["R010", "R007", "R006"]
