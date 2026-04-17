from __future__ import annotations

from collections.abc import Iterable

from codemint.rules.builtin import DiagnosisRule


class RuleEngine:
    def __init__(self, rules: Iterable[DiagnosisRule]) -> None:
        self._rules = tuple(sorted(rules, key=lambda rule: rule.priority))

    def match(self, text: str) -> DiagnosisRule | None:
        for rule in self._rules:
            if rule.pattern.search(text):
                return rule
        return None
