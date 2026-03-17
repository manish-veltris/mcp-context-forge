# -*- coding: utf-8 -*-
"""Tests for the experimental Rust LLM Gateway proxy helper."""

# Standard
from types import SimpleNamespace
from unittest.mock import AsyncMock

# Third-Party
import httpx
import pytest

# First-Party
from mcpgateway.llm_schemas import ChatCompletionRequest, ChatMessage
from mcpgateway.services.llm_provider_service import LLMModelNotFoundError
from mcpgateway.services.llm_proxy_service import LLMProxyRequestError
from mcpgateway.services.rust_llm_gateway_proxy import RustLLMGatewayProxy


class DummyStreamResponse:
    def __init__(self, lines, status_code=200, json_body=None):
        self._lines = lines
        self.status_code = status_code
        self._json_body = json_body or {}
        self.text = ""

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def json(self):
        return self._json_body


class DummyStreamContext:
    def __init__(self, response):
        self.response = response
        self.closed = False

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True
        return False


@pytest.mark.asyncio
async def test_chat_completion_success(monkeypatch: pytest.MonkeyPatch):
    helper = RustLLMGatewayProxy()
    request = ChatCompletionRequest(model="gpt-4", messages=[ChatMessage(role="user", content="hi")])

    response = httpx.Response(
        status_code=200,
        json={
            "id": "resp",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        request=httpx.Request("POST", "http://rust/v1/chat/completions"),
    )

    client = SimpleNamespace(post=AsyncMock(return_value=response))
    monkeypatch.setattr("mcpgateway.services.rust_llm_gateway_proxy.get_http_client", AsyncMock(return_value=client))

    result = await helper.chat_completion(request)

    assert result.id == "resp"
    assert result.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_prepare_chat_completion_stream_success(monkeypatch: pytest.MonkeyPatch):
    helper = RustLLMGatewayProxy()
    request = ChatCompletionRequest(model="gpt-4", messages=[ChatMessage(role="user", content="hi")], stream=True)

    response = DummyStreamResponse(["data: {\"choices\": []}", "data: [DONE]"])
    context = DummyStreamContext(response)
    client = SimpleNamespace(stream=lambda *args, **kwargs: context)
    monkeypatch.setattr("mcpgateway.services.rust_llm_gateway_proxy.get_http_client", AsyncMock(return_value=client))

    stream = await helper.prepare_chat_completion_stream(request)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == ["data: {\"choices\": []}\n\n", "data: [DONE]\n\n"]
    assert context.closed is True


@pytest.mark.asyncio
async def test_prepare_chat_completion_stream_maps_404(monkeypatch: pytest.MonkeyPatch):
    helper = RustLLMGatewayProxy()
    request = ChatCompletionRequest(model="missing", messages=[ChatMessage(role="user", content="hi")], stream=True)

    response = DummyStreamResponse([], status_code=404, json_body={"detail": "Model not found: missing"})
    context = DummyStreamContext(response)
    client = SimpleNamespace(stream=lambda *args, **kwargs: context)
    monkeypatch.setattr("mcpgateway.services.rust_llm_gateway_proxy.get_http_client", AsyncMock(return_value=client))

    with pytest.raises(LLMModelNotFoundError):
        await helper.prepare_chat_completion_stream(request)

    assert context.closed is True


@pytest.mark.asyncio
async def test_list_models_unavailable(monkeypatch: pytest.MonkeyPatch):
    helper = RustLLMGatewayProxy()
    client = SimpleNamespace(get=AsyncMock(side_effect=httpx.ConnectError("boom")))
    monkeypatch.setattr("mcpgateway.services.rust_llm_gateway_proxy.get_http_client", AsyncMock(return_value=client))

    with pytest.raises(LLMProxyRequestError, match="unavailable"):
        await helper.list_models()
