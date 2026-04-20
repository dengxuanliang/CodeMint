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
    with pytest.raises(ValueError, match="difficulty_levels|medium|hard"):
        SynthesizeConfig(difficulty_levels=["easy", "hard"])


def test_invalid_difficulty_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="difficulty_levels"):
        SynthesizeConfig(difficulty_levels=["medium", "expert"])


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "codemint.yaml"
    path.write_text("model:\n  unknown_field: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown_field"):
        load_config(path)


def test_load_config_rejects_invalid_retry_backoff(tmp_path: Path) -> None:
    path = tmp_path / "codemint.yaml"
    path.write_text('model:\n  retry_backoff: "jitter"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="retry_backoff"):
        load_config(path)


def test_load_config_rejects_invalid_custom_pattern_choices(tmp_path: Path) -> None:
    path = tmp_path / "codemint.yaml"
    path.write_text(
        (
            "rules:\n"
            "  custom_patterns:\n"
            "    - name: custom_timeout\n"
            '      pattern: "TimeLimitExceeded|TLE"\n'
            '      fault_type: "logic"\n'
            '      sub_tag: "time_complexity_exceeded"\n'
            '      severity: "critical"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fault_type|severity"):
        load_config(path)


def test_load_config_rejects_invalid_severity_override_values(tmp_path: Path) -> None:
    path = tmp_path / "codemint.yaml"
    path.write_text(
        (
            "rules:\n"
            "  severity_overrides:\n"
            '    missing_import: "critical"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="severity_overrides"):
        load_config(path)


def test_load_config_rejects_invalid_difficulty_distribution(tmp_path: Path) -> None:
    path = tmp_path / "codemint.yaml"
    path.write_text('synthesize:\n  difficulty_distribution: "random"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="difficulty_distribution"):
        load_config(path)


def test_load_config_parses_prompt_override_directory(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "custom-prompts"
    prompt_dir.mkdir()
    path = tmp_path / "codemint.yaml"
    path.write_text(
        f'prompts:\n  override_dir: "{prompt_dir}"\n',
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.prompts.override_dir == prompt_dir
