from __future__ import annotations

from codemint.prompts.registry import PromptTemplate, load_prompt


def test_prompt_registry_extracts_version_header() -> None:
    prompt = load_prompt("diagnose_deep_analysis")

    assert isinstance(prompt, PromptTemplate)
    assert prompt.name == "diagnose_deep_analysis"
    assert prompt.version == "v1"
    assert "deep analysis" in prompt.text.lower()

