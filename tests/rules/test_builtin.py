from codemint.rules.builtin import default_rules


EXPECTED_RULE_METADATA = {
    "R007": ("modeling", "time_limit_exceeded", False),
    "R006": ("modeling", "recursion_depth_exceeded", False),
    "R009": ("surface", "compilation_error", False),
    "R001": ("surface", "syntax_error", False),
    "R002": ("surface", "undefined_variable", False),
    "R003": ("surface", "missing_import", False),
    "R004": ("implementation", "argument_mismatch", False),
    "R008": ("implementation", "missing_return_or_empty_output", False),
    "R012": ("edge_handling", "invalid_math_edge_case", False),
    "R005": ("edge_handling", "missing_bounds_or_key_check", False),
    "R011": ("surface", "output_format_mismatch", False),
    "R010": ("modeling", "assertion_failure", True),
}


def test_default_rules_include_all_expected_builtins() -> None:
    rules = default_rules()

    assert [rule.rule_id for rule in rules] == [
        "R007",
        "R006",
        "R009",
        "R001",
        "R002",
        "R003",
        "R004",
        "R008",
        "R012",
        "R005",
        "R011",
        "R010",
    ]


def test_builtin_rules_are_sorted_by_priority() -> None:
    rules = default_rules()

    assert [rule.priority for rule in rules] == list(range(1, 13))


def test_builtin_rules_have_expected_fault_types_sub_tags_and_analysis_flags() -> None:
    rules = {rule.rule_id: rule for rule in default_rules()}

    assert {
        rule_id: (rule.fault_type, rule.sub_tag, rule.requires_model_analysis)
        for rule_id, rule in rules.items()
    } == EXPECTED_RULE_METADATA


def test_r010_requires_model_analysis_but_normal_rules_do_not() -> None:
    rules = {rule.rule_id: rule for rule in default_rules()}

    assert rules["R010"].requires_model_analysis is True
    assert rules["R007"].requires_model_analysis is False
    assert rules["R001"].requires_model_analysis is False


def test_builtin_rules_cover_representative_language_variants() -> None:
    rules = {rule.rule_id: rule for rule in default_rules()}

    assert rules["R009"].pattern.search("error: expected ';' before '}' token")
    assert rules["R009"].pattern.search("Main.java:3: error: cannot find symbol")
    assert rules["R005"].pattern.search("ArrayIndexOutOfBoundsException")
    assert rules["R005"].pattern.search("panic: runtime error: index out of range [2] with length 2")
    assert rules["R012"].pattern.search("ZeroDivisionError: division by zero")
    assert rules["R012"].pattern.search("ValueError: math domain error")
