from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
VERSION_PREFIX = "Version:"


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    version: str
    text: str

    @property
    def template(self) -> str:
        return self.text


def load_prompt(name: str) -> PromptTemplate:
    safe_name = _validate_prompt_name(name)
    path = (PROMPTS_DIR / f"{safe_name}.txt").resolve()
    raw_text = path.read_text(encoding="utf-8").strip()
    lines = raw_text.splitlines()
    if not lines or not lines[0].startswith(VERSION_PREFIX):
        raise ValueError(f"Prompt {name!r} is missing a version header")

    version = lines[0].split(":", 1)[1].strip()
    body = "\n".join(lines[1:]).lstrip()
    return PromptTemplate(name=name, version=version, text=body)


def _validate_prompt_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError("Invalid prompt name")

    candidate = (PROMPTS_DIR / f"{name}.txt").resolve()
    prompts_root = PROMPTS_DIR.resolve()
    if prompts_root not in candidate.parents:
        raise ValueError("Invalid prompt name")
    return name
