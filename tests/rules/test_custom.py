import pytest

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


def test_disabled_rules_remove_custom_rules_by_name() -> None:
    config = RulesConfig(
        custom_patterns=[
            CustomPatternConfig(
                name="custom_timeout",
                pattern="Time Limit Exceeded",
                fault_type="implementation",
                sub_tag="time_complexity_exceeded",
                severity="high",
            )
        ],
        disabled_rules=["custom_timeout"],
    )

    rules = build_rules(config)

    assert "custom_timeout" not in [rule.rule_id for rule in rules]


def test_severity_override_applies_by_sub_tag() -> None:
    config = RulesConfig(severity_overrides={"missing_import": "low"})

    rules = build_rules(config)
    by_id = {rule.rule_id: rule for rule in rules}

    assert by_id["R003"].severity == "low"


def test_custom_rule_severity_override_applies_by_rule_name() -> None:
    config = RulesConfig(
        custom_patterns=[
            CustomPatternConfig(
                name="custom_timeout",
                pattern="Time Limit Exceeded",
                fault_type="implementation",
                sub_tag="time_complexity_exceeded",
                severity="high",
            )
        ],
        severity_overrides={"custom_timeout": "low"},
    )

    rules = build_rules(config)
    by_id = {rule.rule_id: rule for rule in rules}

    assert by_id["custom_timeout"].severity == "low"


def test_rule_priority_reorders_rules_without_losing_others() -> None:
    config = RulesConfig(rule_priority=["R010", "R007"])

    rules = build_rules(config)

    assert [rule.rule_id for rule in rules[:3]] == ["R010", "R007", "R006"]


def test_empty_custom_rule_name_is_rejected() -> None:
    config = RulesConfig(
        custom_patterns=[
            CustomPatternConfig(
                name="  ",
                pattern="Time Limit Exceeded",
                fault_type="implementation",
                sub_tag="time_complexity_exceeded",
                severity="high",
            )
        ]
    )

    with pytest.raises(ValueError, match="custom rule name"):
        build_rules(config)


def test_duplicate_custom_rule_names_are_rejected() -> None:
    config = RulesConfig(
        custom_patterns=[
            CustomPatternConfig(
                name="custom_timeout",
                pattern="Time Limit Exceeded",
                fault_type="implementation",
                sub_tag="time_complexity_exceeded",
                severity="high",
            ),
            CustomPatternConfig(
                name=" custom_timeout ",
                pattern="TLE",
                fault_type="modeling",
                sub_tag="time_limit_exceeded",
                severity="high",
            ),
        ]
    )

    with pytest.raises(ValueError, match="duplicate custom rule"):
        build_rules(config)


def test_custom_rule_name_colliding_with_builtin_rule_id_is_rejected() -> None:
    config = RulesConfig(
        custom_patterns=[
            CustomPatternConfig(
                name="R007",
                pattern="custom timeout",
                fault_type="implementation",
                sub_tag="time_complexity_exceeded",
                severity="high",
            )
        ]
    )

    with pytest.raises(ValueError, match="built-in rule"):
        build_rules(config)


def test_duplicate_rule_priority_ids_are_rejected() -> None:
    config = RulesConfig(rule_priority=["R007", "R007"])

    with pytest.raises(ValueError, match="duplicate rule_priority"):
        build_rules(config)


def test_unknown_rule_priority_ids_are_rejected() -> None:
    config = RulesConfig(rule_priority=["R007", "unknown_rule"])

    with pytest.raises(ValueError, match="unknown rule_priority"):
        build_rules(config)
