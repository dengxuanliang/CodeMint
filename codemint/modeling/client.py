from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import httpx

from codemint.config import ModelConfig


class ModelClient:
    def __init__(
        self,
        config: ModelConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if config.max_retries <= 0:
            raise ValueError("max_retries must be greater than 0")

        self._config = config
        self._sleeper = sleeper or time.sleep
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.timeout),
            transport=transport,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                response = self._client.post(
                    _completion_url(self._config.base_url),
                    json=self._build_payload(system_prompt, user_prompt),
                    headers=self._build_headers(),
                )
                response.raise_for_status()
                payload = response.json()
                return _extract_text(payload)
            except httpx.HTTPError as error:
                last_error = error
                if not _is_retryable_error(error) or attempt == self._config.max_retries:
                    break
                self._sleeper(_retry_delay_seconds(attempt, self._config.retry_backoff))

        assert last_error is not None
        raise last_error

    def _build_payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "model": self._config.analysis_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload["choices"]
    return choices[0]["message"]["content"]


def _retry_delay_seconds(attempt: int, backoff: str) -> float:
    if backoff == "exponential":
        return float(2 ** (attempt - 1))
    raise ValueError(f"Unsupported retry_backoff: {backoff}")


def _is_retryable_error(error: httpx.HTTPError) -> bool:
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code == 429 or status_code >= 500
    return False


def _completion_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"
