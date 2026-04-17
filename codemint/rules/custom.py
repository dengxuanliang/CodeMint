from __future__ import annotations

import re
from dataclasses import replace

from codemint.config import CustomPatternConfig, RulesConfig
from codemint.models.diagnosis import Severity
from codemint.rules.builtin import DiagnosisRule, default_rules


def _custom_rule_id(name: str) -> str:
    return name.strip()


def _custom_priority(index: int) -> int:
    return 1000 + index


def build_rules(
    config: RulesConfig | None = None,
    *,
    custom_patterns: list[CustomPatternConfig] | None = None,
    disabled_rules: list[str] | None = None,
    severity_overrides: dict[str, Severity] | None = None,
    rule_priority: list[str] | None = None,
) -> list[DiagnosisRule]:
    rules_config = _rules_config(
        config,
        custom_patterns=custom_patterns,
        disabled_rules=disabled_rules,
        severity_overrides=severity_overrides,
        rule_priority=rule_priority,
    )
    disabled = set(rules_config.disabled_rules)
    rules = [rule for rule in default_rules() if rule.rule_id not in disabled]
    built_in_ids = {rule.rule_id for rule in default_rules()}

    _validate_custom_rule_ids(rules_config, built_in_ids)

    rules = [_apply_severity_override(rule, rules_config) for rule in rules]
    rules.extend(_build_custom_rules(rules_config))
    _validate_rule_priority(rules_config.rule_priority, {rule.rule_id for rule in rules})
    return _sort_rules(rules, rules_config.rule_priority)


def _rules_config(
    config: RulesConfig | None,
    *,
    custom_patterns: list[CustomPatternConfig] | None,
    disabled_rules: list[str] | None,
    severity_overrides: dict[str, Severity] | None,
    rule_priority: list[str] | None,
) -> RulesConfig:
    base = config or RulesConfig()
    return RulesConfig(
        custom_patterns=custom_patterns if custom_patterns is not None else base.custom_patterns,
        disabled_rules=disabled_rules if disabled_rules is not None else base.disabled_rules,
        severity_overrides=(
            severity_overrides if severity_overrides is not None else base.severity_overrides
        ),
        rule_priority=rule_priority if rule_priority is not None else base.rule_priority,
    )


def _apply_severity_override(rule: DiagnosisRule, config: RulesConfig) -> DiagnosisRule:
    severity = config.severity_overrides.get(rule.sub_tag)
    severity = config.severity_overrides.get(rule.rule_id, severity)
    if severity is None:
        return rule
    return replace(rule, severity=severity)


def _validate_custom_rule_ids(config: RulesConfig, built_in_ids: set[str]) -> None:
    seen: set[str] = set()
    for custom in config.custom_patterns:
        rule_id = _custom_rule_id(custom.name)
        if not rule_id:
            raise ValueError("custom rule name must not be empty")
        if rule_id in seen:
            raise ValueError(f"duplicate custom rule name: {rule_id}")
        if rule_id in built_in_ids:
            raise ValueError(f"custom rule name collides with built-in rule ID: {rule_id}")
        seen.add(rule_id)


def _validate_rule_priority(rule_priority: list[str], known_rule_ids: set[str]) -> None:
    seen: set[str] = set()
    for rule_id in rule_priority:
        if rule_id in seen:
            raise ValueError(f"duplicate rule_priority ID: {rule_id}")
        seen.add(rule_id)

    unknown = [rule_id for rule_id in rule_priority if rule_id not in known_rule_ids]
    if unknown:
        raise ValueError(f"unknown rule_priority ID: {unknown[0]}")


def _build_custom_rules(config: RulesConfig) -> list[DiagnosisRule]:
    rules: list[DiagnosisRule] = []
    for index, custom in enumerate(config.custom_patterns, start=1):
        rule_id = _custom_rule_id(custom.name)
        if rule_id in config.disabled_rules:
            continue
        severity = config.severity_overrides.get(custom.sub_tag, custom.severity)
        severity = config.severity_overrides.get(rule_id, severity)
        rules.append(
            DiagnosisRule(
                rule_id=rule_id,
                pattern=re.compile(custom.pattern, re.IGNORECASE | re.MULTILINE),
                fault_type=custom.fault_type,
                sub_tag=custom.sub_tag,
                severity=severity,
                priority=_custom_priority(index),
            )
        )
    return rules


def _sort_rules(rules: list[DiagnosisRule], priority_order: list[str]) -> list[DiagnosisRule]:
    explicit_order = {rule_id: index for index, rule_id in enumerate(priority_order)}
    ordered = sorted(
        rules,
        key=lambda rule: (
            explicit_order.get(rule.rule_id, len(explicit_order)),
            rule.priority,
            rule.rule_id,
        ),
    )
    return [replace(rule, priority=index) for index, rule in enumerate(ordered, start=1)]
