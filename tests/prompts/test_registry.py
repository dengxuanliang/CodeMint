from __future__ import annotations

from pathlib import Path

import pytest

from codemint.prompts import registry
from codemint.prompts.registry import PromptTemplate, load_prompt


def test_prompt_registry_extracts_version_header() -> None:
    prompt = load_prompt("diagnose_deep_analysis")

    assert isinstance(prompt, PromptTemplate)
    assert prompt.name == "diagnose_deep_analysis"
    assert prompt.version == "v1"
    assert "deep analysis" in prompt.text.lower()


@pytest.mark.parametrize("name", ["", "../secret", "nested/prompt", r"nested\prompt"])
def test_prompt_registry_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="prompt name"):
        load_prompt(name)


def test_prompt_registry_requires_version_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "invalid.txt").write_text("Missing header\nbody", encoding="utf-8")
    monkeypatch.setattr(registry, "PROMPTS_DIR", prompt_dir)

    with pytest.raises(ValueError, match="version header"):
        load_prompt("invalid")
