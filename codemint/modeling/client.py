from __future__ import annotations

from collections.abc import Callable
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
        self._config = config
        self._sleeper = sleeper or (lambda _: None)
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
                    f"{self._config.base_url.rstrip('/')}/chat/completions",
                    json=self._build_payload(system_prompt, user_prompt),
                    headers=self._build_headers(),
                )
                response.raise_for_status()
                payload = response.json()
                return _extract_text(payload)
            except httpx.HTTPError as error:
                last_error = error
                if attempt == self._config.max_retries:
                    break
                self._sleeper(_retry_delay_seconds(attempt))

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


def _retry_delay_seconds(attempt: int) -> float:
    return float(2 ** (attempt - 1))
