from pathlib import Path

import pytest

from codemint.config import SynthesizeConfig, load_config


def test_load_config_expands_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    path = tmp_path / "codemint.yaml"
    path.write_text('model:\n  api_key: "${OPENAI_API_KEY}"\n', encoding="utf-8")

    config = load_config(path)

    assert config.model.api_key == "secret"


def test_load_config_parses_typed_defaults(tmp_path: Path) -> None:
    path = tmp_path / "codemint.yaml"
    path.write_text("{}", encoding="utf-8")

    config = load_config(path)

    assert config.model.base_url == "https://api.openai.com/v1"
    assert config.aggregate.verification_level == "auto"
    assert config.synthesize.difficulty_levels == ["medium", "hard"]


def test_easy_difficulty_is_rejected() -> None:
    with pytest.raises(ValueError, match="easy is not allowed"):
        SynthesizeConfig(difficulty_levels=["easy", "hard"])
