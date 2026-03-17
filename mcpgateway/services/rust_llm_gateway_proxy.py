# -*- coding: utf-8 -*-
"""Experimental Rust LLM Gateway proxy helper."""

# Standard
from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, AsyncGenerator

# Third-Party
import httpx
import orjson

# First-Party
from mcpgateway.config import settings
from mcpgateway.llm_schemas import ChatCompletionRequest, ChatCompletionResponse
from mcpgateway.services.http_client_service import get_http_client
from mcpgateway.services.llm_provider_service import LLMModelNotFoundError, LLMProviderNotFoundError
from mcpgateway.services.llm_proxy_service import LLMProxyAuthError, LLMProxyRequestError

_INTERNAL_FORWARD_HEADER = "x-forwarded-internally"
_INTERNAL_SECRET_HEADER = "x-contextforge-internal-secret"  # nosec B105 - header name, not a secret value


class RustLLMGatewayProxy:
    """Proxy helper for the experimental Rust LLM Gateway sidecar."""

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Send a non-streaming chat completion request to the Rust sidecar.

        Args:
            request: OpenAI-compatible chat completion request payload.

        Returns:
            The validated chat completion response from the Rust sidecar.

        Raises:
            LLMProxyAuthError: If the Rust sidecar rejects authentication.
            LLMModelNotFoundError: If the requested model is unknown.
            LLMProviderNotFoundError: If the configured provider is unavailable.
            LLMProxyRequestError: If the Rust sidecar is unavailable or returns
                an error response.
        """
        client = await get_http_client()
        try:
            response = await client.post(
                self._build_url("/v1/chat/completions"),
                json=request.model_dump(exclude_none=True),
                headers=self._build_headers(),
                timeout=httpx.Timeout(settings.experimental_rust_llm_gateway_timeout_seconds),
            )
        except httpx.HTTPError as exc:
            raise LLMProxyRequestError("Experimental Rust LLM Gateway unavailable") from exc

        self._raise_for_response(response)
        return ChatCompletionResponse.model_validate(response.json())

    async def prepare_chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        """Open a streaming request to the Rust sidecar and return an SSE generator.

        Args:
            request: OpenAI-compatible chat completion request payload.

        Returns:
            An async generator that yields SSE-formatted stream chunks.

        Raises:
            LLMProxyAuthError: If the Rust sidecar rejects authentication.
            LLMModelNotFoundError: If the requested model is unknown.
            LLMProviderNotFoundError: If the configured provider is unavailable.
            LLMProxyRequestError: If the Rust sidecar is unavailable or rejects
                the request.
        """
        client = await get_http_client()
        stream_context = client.stream(
            "POST",
            self._build_url("/v1/chat/completions"),
            json=request.model_dump(exclude_none=True),
            headers=self._build_headers(),
            timeout=httpx.Timeout(settings.experimental_rust_llm_gateway_timeout_seconds),
        )
        exit_stack = AsyncExitStack()
        try:
            response = await exit_stack.enter_async_context(stream_context)
        except httpx.HTTPError as exc:
            raise LLMProxyRequestError("Experimental Rust LLM Gateway unavailable") from exc

        try:
            self._raise_for_response(response)
        except (LLMProxyAuthError, LLMModelNotFoundError, LLMProviderNotFoundError, LLMProxyRequestError):
            await exit_stack.aclose()
            raise

        async def _stream() -> AsyncGenerator[str, None]:
            try:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    yield f"{line}\n\n"
            finally:
                await exit_stack.aclose()

        return _stream()

    async def list_models(self) -> dict[str, Any]:
        """List gateway-visible models through the Rust sidecar.

        Returns:
            The OpenAI-compatible model list payload from the Rust sidecar.

        Raises:
            LLMProxyAuthError: If the Rust sidecar rejects authentication.
            LLMModelNotFoundError: If the requested model is unknown.
            LLMProviderNotFoundError: If the configured provider is unavailable.
            LLMProxyRequestError: If the Rust sidecar is unavailable or returns
                an error response.
        """
        client = await get_http_client()
        try:
            response = await client.get(
                self._build_url("/v1/models"),
                headers=self._build_headers(),
                timeout=httpx.Timeout(settings.experimental_rust_llm_gateway_timeout_seconds),
            )
        except httpx.HTTPError as exc:
            raise LLMProxyRequestError("Experimental Rust LLM Gateway unavailable") from exc

        self._raise_for_response(response)
        return response.json()

    def _build_url(self, path: str) -> str:
        """Build a sidecar URL for a relative API path.

        Args:
            path: Relative API path.

        Returns:
            Fully qualified sidecar URL.
        """
        base = settings.experimental_rust_llm_gateway_url.rstrip("/")
        return f"{base}{path}"

    def _build_headers(self) -> dict[str, str]:
        """Build trusted runtime headers for sidecar requests.

        Returns:
            Headers required for trusted internal sidecar communication.
        """
        headers = {
            _INTERNAL_FORWARD_HEADER: "true",
        }
        if settings.experimental_rust_llm_gateway_internal_secret:
            headers[_INTERNAL_SECRET_HEADER] = settings.experimental_rust_llm_gateway_internal_secret
        return headers

    def _raise_for_response(self, response: httpx.Response) -> None:
        """Raise gateway-specific exceptions for HTTP error responses.

        Args:
            response: HTTP response returned by the Rust sidecar.

        Raises:
            LLMProxyAuthError: If the sidecar returned ``401``.
            LLMModelNotFoundError: If the requested model was not found.
            LLMProviderNotFoundError: If the underlying provider was not found.
            LLMProxyRequestError: For all other failing responses.
        """
        if response.status_code < 400:
            return

        detail = self._extract_detail(response)
        if response.status_code == 401:
            raise LLMProxyAuthError(detail)
        if response.status_code == 404:
            if "provider" in detail.lower():
                raise LLMProviderNotFoundError(detail)
            raise LLMModelNotFoundError(detail)
        raise LLMProxyRequestError(detail)

    def _extract_detail(self, response: httpx.Response) -> str:
        """Extract the best available error detail from a sidecar response.

        Args:
            response: HTTP response returned by the Rust sidecar.

        Returns:
            A human-readable error detail string.
        """
        try:
            data = response.json()
        except ValueError:
            return response.text or f"Rust LLM Gateway request failed: {response.status_code}"

        if isinstance(data, dict):
            detail = data.get("detail")
            if isinstance(detail, str) and detail:
                return detail
            error = data.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            try:
                return orjson.dumps(data).decode("utf-8")
            except TypeError:
                pass
        return response.text or f"Rust LLM Gateway request failed: {response.status_code}"
