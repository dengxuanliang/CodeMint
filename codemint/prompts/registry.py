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
    path = PROMPTS_DIR / f"{name}.txt"
    raw_text = path.read_text(encoding="utf-8").strip()
    lines = raw_text.splitlines()
    if not lines or not lines[0].startswith(VERSION_PREFIX):
        raise ValueError(f"Prompt {name!r} is missing a version header")

    version = lines[0].split(":", 1)[1].strip()
    body = "\n".join(lines[1:]).lstrip()
    return PromptTemplate(name=name, version=version, text=body)

