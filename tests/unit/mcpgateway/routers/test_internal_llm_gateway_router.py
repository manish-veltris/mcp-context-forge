# -*- coding: utf-8 -*-
"""Tests for trusted internal Rust LLM Gateway endpoints."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third-Party
import pytest
from fastapi import HTTPException

# First-Party
from mcpgateway.llm_schemas import ResolveChatCompletionTargetRequest
from mcpgateway.routers import internal_llm_gateway_router
from mcpgateway.services.llm_proxy_service import LLMProxyRequestError


def _trusted_request(headers=None, host="127.0.0.1"):
    return SimpleNamespace(
        client=SimpleNamespace(host=host),
        headers={"x-forwarded-internally": "true", **(headers or {})},
    )


def test_require_trusted_runtime_request_rejects_non_loopback():
    request = _trusted_request(host="10.0.0.10")

    with pytest.raises(HTTPException) as excinfo:
        internal_llm_gateway_router._require_trusted_runtime_request(request)

    assert excinfo.value.status_code == 403


def test_require_trusted_runtime_request_rejects_bad_secret(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(internal_llm_gateway_router.settings, "experimental_rust_llm_gateway_internal_secret", "expected")
    request = _trusted_request(headers={"x-contextforge-internal-secret": "wrong"})

    with pytest.raises(HTTPException) as excinfo:
        internal_llm_gateway_router._require_trusted_runtime_request(request)

    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_list_models(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(internal_llm_gateway_router.settings, "experimental_rust_llm_gateway_internal_secret", None)
    monkeypatch.setattr(
        internal_llm_gateway_router.llm_provider_service,
        "get_gateway_models",
        lambda _db: [
            SimpleNamespace(model_id="gpt-4", provider_name="OpenAI"),
            SimpleNamespace(model_id="claude-3", provider_name="Anthropic"),
        ],
    )

    result = await internal_llm_gateway_router.internal_list_models(request=_trusted_request(), db=MagicMock())

    assert result["object"] == "list"
    assert result["data"][0]["id"] == "gpt-4"
    assert result["data"][1]["owned_by"] == "Anthropic"


@pytest.mark.asyncio
async def test_internal_resolve_chat_target_maps_proxy_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(internal_llm_gateway_router.settings, "experimental_rust_llm_gateway_internal_secret", None)
    monkeypatch.setattr(
        internal_llm_gateway_router.llm_proxy_service,
        "resolve_chat_completion_target",
        lambda _db, _model: (_ for _ in ()).throw(LLMProxyRequestError("invalid target")),
    )

    with pytest.raises(HTTPException) as excinfo:
        await internal_llm_gateway_router.internal_resolve_chat_target(
            payload=ResolveChatCompletionTargetRequest(model="gpt-4"),
            request=_trusted_request(),
            db=MagicMock(),
        )

    assert excinfo.value.status_code == 502
