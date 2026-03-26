# Standard
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Third-Party
import orjson
import pytest
from sqlalchemy.orm import Session

# First-Party
from mcpgateway.schemas import TextContent, ToolRead, ToolResult
from mcpgateway.services.tool_service import ToolService


@pytest.mark.asyncio
async def test_invoke_tool_with_flattened_arguments():
    """
    Test that invoke_tool correctly unflattens dot-notation arguments
    before sending them to the upstream REST API.
    """
    db = MagicMock(spec=Session)
    service = ToolService()

    mock_tool = SimpleNamespace(id="123", name="test_tool", enabled=True, reachable=True, gateway=MagicMock())

    service._load_invocable_tools = MagicMock(return_value=[mock_tool])

    # Mock cache payload builder to return required fields
    service._build_tool_cache_payload = MagicMock(
        return_value={
            "tool": {
                "id": "123",
                "name": "test_tool",
                "original_name": "test_tool",
                "integration_type": "REST",
                "request_type": "POST",
                "url": "https://api.example.com/data",
                "enabled": True,
                "reachable": True,
                "headers": {},
                "auth_type": None,
                "auth_value": None,
                "jsonpath_filter": None,
                "output_schema": {"type": "object"},
                "timeout_ms": 5000,
                "gateway_id": "gw1",
            },
            "gateway": {"id": "gw1", "url": "https://api.example.com", "auth_type": None, "gateway_mode": "gateway"},
        }
    )

    # Mock HTTP client
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.text = '{"status": "ok"}'
    mock_response.raise_for_status = MagicMock()

    service._http_client.request = AsyncMock(return_value=mock_response)

    # Flattened arguments (dot-notation)
    arguments = {"user.name": "Alice", "user.details.age": 30, "id": 101}

    with (
        patch.object(ToolService, "_check_tool_access", return_value=True),
        patch("mcpgateway.services.tool_service.current_trace_id") as mock_trace,
        patch("mcpgateway.services.tool_service.fresh_db_session"),
        patch("mcpgateway.services.tool_service.get_correlation_id", return_value="test_corr_id"),
    ):
        mock_trace.get.return_value = None
        await service.invoke_tool(db=db, name="test_tool", arguments=arguments, user_email="admin@example.com")

    # Verify that the HTTP request used UNFLATTENED payload
    service._http_client.request.assert_called_once()
    _, kwargs = service._http_client.request.call_args
    sent_payload = kwargs.get("json")

    assert sent_payload == {"user": {"name": "Alice", "details": {"age": 30}}, "id": 101}


@pytest.mark.asyncio
async def test_invoke_tool_flattens_response():
    """
    Test that invoke_tool correctly flattens nested response data
    before returning it to the caller.
    """
    db = MagicMock(spec=Session)
    service = ToolService()

    mock_tool = SimpleNamespace(id="123", name="test_tool", enabled=True, reachable=True, gateway=MagicMock())
    service._load_invocable_tools = MagicMock(return_value=[mock_tool])

    service._build_tool_cache_payload = MagicMock(
        return_value={
            "tool": {
                "id": "123",
                "name": "test_tool",
                "integration_type": "REST",
                "request_type": "GET",
                "url": "https://api.example.com/user",
                "enabled": True,
                "reachable": True,
                "headers": {},
                "auth_type": None,
                "jsonpath_filter": None,
                "output_schema": {"type": "object"},
                "timeout_ms": 5000,
                "gateway_id": "gw1",
            },
            "gateway": {"id": "gw1", "url": "https://api.example.com", "gateway_mode": "gateway"},
        }
    )

    # Mock HTTP response with nested data
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"user": {"name": "Bob", "metadata": {"last_login": "2023-01-01"}}}
    mock_response.raise_for_status = MagicMock()
    service._http_client.get = AsyncMock(return_value=mock_response)

    with (
        patch.object(ToolService, "_check_tool_access", return_value=True),
        patch("mcpgateway.services.tool_service.current_trace_id") as mock_trace,
        patch("mcpgateway.services.tool_service.fresh_db_session"),
        patch("mcpgateway.services.tool_service.get_correlation_id", return_value="test_corr_id"),
    ):
        mock_trace.get.return_value = None
        result = await service.invoke_tool(db=db, name="test_tool", arguments={}, user_email="admin@example.com")

    # Verify the result content is flattened
    content_text = result.content[0].text
    flattened_data = orjson.loads(content_text)

    assert flattened_data == {"user.name": "Bob", "user.metadata.last_login": "2023-01-01"}
