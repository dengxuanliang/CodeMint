from __future__ import annotations

import json
import time

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


def test_model_client_does_not_retry_non_retryable_401() -> None:
    attempts = {"count": 0}
    request = httpx.Request("POST", "https://example.test/chat/completions")

    def handler(_: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(401, request=request, json={"error": "unauthorized"})

    client = ModelClient(
        ModelConfig(base_url="https://example.test", max_retries=3),
        transport=httpx.MockTransport(handler),
        sleeper=lambda _: None,
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.complete("system prompt", "user prompt")

    assert attempts["count"] == 1


def test_model_client_retries_retryable_429_response() -> None:
    attempts = {"count": 0}
    request = httpx.Request("POST", "https://example.test/chat/completions")

    def handler(_: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(429, request=request, json={"error": "rate_limited"})
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "final answer"}}]},
        )

    client = ModelClient(
        ModelConfig(base_url="https://example.test", max_retries=2),
        transport=httpx.MockTransport(handler),
        sleeper=lambda _: None,
    )

    assert client.complete("system prompt", "user prompt") == "final answer"
    assert attempts["count"] == 2


def test_model_client_uses_exponential_backoff_with_injected_sleeper() -> None:
    delays: list[float] = []
    attempts = {"count": 0}

    def fake_post(
        self: httpx.Client,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> DummyResponse:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.ReadTimeout("slow")
        return DummyResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(httpx.Client, "post", fake_post)
    try:
        client = ModelClient(
            ModelConfig(
                base_url="https://example.test",
                max_retries=3,
                retry_backoff="exponential",
            ),
            sleeper=delays.append,
        )

        assert client.complete("system prompt", "user prompt") == "ok"
    finally:
        monkeypatch.undo()

    assert delays == [1.0, 2.0]


def test_model_client_uses_time_sleep_by_default() -> None:
    client = ModelClient(ModelConfig(base_url="https://example.test"))

    assert client._sleeper is time.sleep


def test_model_client_rejects_non_positive_max_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        ModelClient(ModelConfig(base_url="https://example.test", max_retries=0))


def test_model_client_does_not_duplicate_chat_completions_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {"url": None}

    def fake_post(
        self: httpx.Client,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> DummyResponse:
        seen["url"] = url
        return DummyResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    client = ModelClient(
        ModelConfig(
            base_url="https://example.test/chat/completions",
            analysis_model="gpt-test",
        ),
        sleeper=lambda _: None,
    )

    assert client.complete("system prompt", "user prompt") == "ok"
    assert seen["url"] == "https://example.test/chat/completions"


def test_model_client_includes_determinism_fields_in_payload() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}}]},
        )

    client = ModelClient(
        ModelConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            analysis_model="gpt-test",
            temperature=0,
            seed=7,
        ),
        transport=httpx.MockTransport(handler),
    )

    client.complete("system", "user")

    assert captured["temperature"] == 0
    assert captured["seed"] == 7
