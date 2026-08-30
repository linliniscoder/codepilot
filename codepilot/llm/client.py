from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8001/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_TIMEOUT = 60.0
DEFAULT_CONTEXT_WINDOW = 4096


class VLLMClientError(RuntimeError):
    """Base exception for vLLM client failures."""


class ServerUnavailableError(VLLMClientError):
    """Raised when the vLLM server cannot be reached or is unavailable."""


class LLMTimeoutError(VLLMClientError):
    """Raised when the vLLM request exceeds its timeout."""


class EmptyResponseError(VLLMClientError):
    """Raised when vLLM returns no usable assistant content."""


class VLLMClient:
    """Thin client for a vLLM server exposing the OpenAI-compatible API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        timeout: float = DEFAULT_TIMEOUT,
        context_window: int | None = None,
        client: OpenAI | None = None,
    ) -> None:
        configured_context = context_window or int(
            os.getenv("CODEPILOT_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW)
        )
        if configured_context < 512:
            raise ValueError("context_window must be at least 512 tokens")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.context_window = configured_context
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Send a chat completion request and return normalized response data."""
        request_started_at = time.perf_counter()
        request_messages = list(messages)
        effective_max_tokens = self._fit_max_tokens(request_messages, max_tokens)
        LOGGER.info(
            "LLM request started: model=%s messages=%d max_tokens=%d",
            self.model,
            len(request_messages),
            effective_max_tokens,
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=request_messages,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )
        except APITimeoutError as exc:
            latency = self._latency(request_started_at)
            LOGGER.exception("LLM request timed out: latency=%.3fs", latency)
            raise LLMTimeoutError(
                f"Timed out while waiting for vLLM after {latency:.3f}s"
            ) from exc
        except APIConnectionError as exc:
            latency = self._latency(request_started_at)
            LOGGER.exception(
                "vLLM server unavailable: url=%s latency=%.3fs",
                self.base_url,
                latency,
            )
            raise ServerUnavailableError(
                f"Could not connect to vLLM server at {self.base_url}"
            ) from exc
        except APIStatusError as exc:
            latency = self._latency(request_started_at)
            if exc.status_code >= 500:
                LOGGER.exception(
                    "vLLM server unavailable: status=%s latency=%.3fs",
                    exc.status_code,
                    latency,
                )
                raise ServerUnavailableError(
                    f"vLLM server returned unavailable status {exc.status_code}"
                ) from exc

            LOGGER.exception(
                "vLLM request failed: status=%s latency=%.3fs",
                exc.status_code,
                latency,
            )
            raise VLLMClientError(
                f"vLLM request failed with status {exc.status_code}"
            ) from exc

        latency = self._latency(request_started_at)
        content = self._extract_content(response)
        if not content:
            LOGGER.error("LLM returned an empty response: latency=%.3fs", latency)
            raise EmptyResponseError("vLLM returned an empty assistant response")

        usage = self._normalize_usage(getattr(response, "usage", None))
        LOGGER.info(
            "LLM request completed: total_tokens=%s prompt_tokens=%s "
            "completion_tokens=%s latency=%.3fs",
            usage.get("total_tokens", "unknown"),
            usage.get("prompt_tokens", "unknown"),
            usage.get("completion_tokens", "unknown"),
            latency,
        )

        return {
            "content": content,
            "usage": usage,
            "latency": latency,
        }

    @staticmethod
    def _latency(request_started_at: float) -> float:
        return time.perf_counter() - request_started_at

    def _fit_max_tokens(
        self,
        messages: Sequence[dict[str, Any]],
        requested_max_tokens: int,
    ) -> int:
        """Keep output inside the server context window without a second request."""
        if requested_max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        input_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
        # This conservative estimate covers both Chinese text and source code.
        estimated_input_tokens = max(1, (len(input_text) + 1) // 2)
        available = self.context_window - estimated_input_tokens - 64
        if available < 64:
            raise VLLMClientError(
                "Prompt is too large for the configured context window "
                f"({self.context_window} tokens)"
            )
        effective = min(requested_max_tokens, available)
        if effective != requested_max_tokens:
            LOGGER.warning(
                "Reducing max_tokens from %d to %d for context window=%d "
                "estimated_input_tokens=%d",
                requested_max_tokens,
                effective,
                self.context_window,
                estimated_input_tokens,
            )
        return max(64, effective)

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            return ""
        return content.strip()

    @staticmethod
    def _normalize_usage(usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return dict(usage)
        if hasattr(usage, "model_dump"):
            return usage.model_dump()

        fields = ("prompt_tokens", "completion_tokens", "total_tokens")
        return {
            field: value
            for field in fields
            if (value := getattr(usage, field, None)) is not None
        }
