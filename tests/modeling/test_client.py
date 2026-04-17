from __future__ import annotations

import httpx
import pytest

from codemint.config import ModelConfig
from codemint.modeling.client import ModelClient


class DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


def test_model_client_retries_once_and_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def fake_post(
        self: httpx.Client,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> DummyResponse:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ConnectError("boom")
        assert url == "https://example.test/chat/completions"
        assert json["model"] == "gpt-test"
        assert headers["Authorization"] == "Bearer secret"
        return DummyResponse(
            {"choices": [{"message": {"content": "final answer"}}]}
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    client = ModelClient(
        ModelConfig(
            base_url="https://example.test",
            api_key="secret",
            analysis_model="gpt-test",
            max_retries=2,
            timeout=7,
        ),
        sleeper=lambda _: None,
    )

    text = client.complete("system prompt", "user prompt")

    assert text == "final answer"
    assert attempts["count"] == 2


def test_model_client_uses_configured_timeout() -> None:
    client = ModelClient(
        ModelConfig(base_url="https://example.test", timeout=9),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )
        ),
    )

    assert client._client.timeout.connect == 9

