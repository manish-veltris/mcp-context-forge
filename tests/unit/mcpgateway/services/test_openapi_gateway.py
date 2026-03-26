# Standard
from unittest.mock import AsyncMock, MagicMock

# Third-Party
from httpx import Response
import pytest

# First-Party
from mcpgateway.schemas import ToolCreate
from mcpgateway.services.gateway_service import GatewayService


@pytest.mark.asyncio
async def test_initialize_gateway_openapi():
    """Test that GatewayService can initialize from an OpenAPI spec and create tools."""
    service = GatewayService()

    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.content = b"""{
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "servers": [{"url": "https://api.example.com/v1"}],
        "paths": {
            "/users": {
                "get": {
                    "operationId": "getUsers",
                    "summary": "Get all users",
                    "parameters": [{"name": "limit", "in": "query", "schema": {"type": "integer"}}]
                },
                "post": {
                    "operationId": "createUser",
                    "summary": "Create a user",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                    "required": ["name"]
                                }
                            }
                        }
                    }
                }
            }
        }
    }"""

    service._http_client.get = AsyncMock(return_value=mock_response)

    capabilities, tools, resources, prompts = await service._initialize_gateway(url="https://api.example.com/openapi.json", transport="OPENAPI")

    assert capabilities.get("openapi") == "3.0.0"
    assert len(tools) == 2

    get_tool = next(t for t in tools if t.name == "getUsers")
    assert get_tool.request_type == "GET"
    assert get_tool.integration_type == "REST"
    assert "limit" in get_tool.input_schema["properties"]
    assert get_tool.url == "https://api.example.com/v1/users"

    post_tool = next(t for t in tools if t.name == "createUser")
    assert post_tool.request_type == "POST"
    assert post_tool.integration_type == "REST"
    assert "name" in post_tool.input_schema["properties"]
    assert "name" in post_tool.input_schema["required"]
    assert post_tool.url == "https://api.example.com/v1/users"


@pytest.mark.asyncio
async def test_initialize_gateway_openapi_fallback_url():
    """Test OpenAPI spec parser fallback when 'servers' is missing."""
    service = GatewayService()

    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.content = b"""{
        "openapi": "3.0.0",
        "paths": {
            "/status": {
                "get": {
                    "operationId": "getStatus"
                }
            }
        }
    }"""

    service._http_client.get = AsyncMock(return_value=mock_response)

    capabilities, tools, resources, prompts = await service._initialize_gateway(url="https://backup.example.com/docs/openapi.json", transport="OPENAPI")

    assert len(tools) == 1
    get_tool = tools[0]

    # Should fallback to the directory of the URL provided
    assert get_tool.url == "https://backup.example.com/docs/status"
    assert get_tool.request_type == "GET"
    assert get_tool.integration_type == "REST"


@pytest.mark.asyncio
async def test_initialize_gateway_swagger_2_0_petstore():
    """Test parsing of a Swagger 2.0 spec with in: body and $ref resolution."""
    service = GatewayService()

    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.content = b"""{
        "swagger": "2.0",
        "info": {"title": "Petstore", "version": "1.0.0"},
        "host": "petstore.swagger.io",
        "basePath": "/v2",
        "paths": {
            "/pet": {
                "post": {
                    "operationId": "addPet",
                    "summary": "Add a new pet to the store",
                    "parameters": [
                        {
                            "in": "body",
                            "name": "body",
                            "description": "Pet object",
                            "required": true,
                            "schema": {
                                "$ref": "#/definitions/Pet"
                            }
                        }
                    ]
                }
            }
        },
        "definitions": {
            "Pet": {
                "type": "object",
                "required": ["name", "photoUrls"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "photoUrls": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "category": {
                        "$ref": "#/definitions/Category"
                    }
                }
            },
            "Category": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"}
                }
            }
        }
    }"""

    service._http_client.get = AsyncMock(return_value=mock_response)

    capabilities, tools, resources, prompts = await service._initialize_gateway(url="https://petstore.swagger.io/v2/swagger.json", transport="OPENAPI")

    assert len(tools) == 1
    post_tool = tools[0]

    # Assert URL fallback mechanism for swagger.json works correctly
    assert post_tool.url == "https://petstore.swagger.io/v2/pet"
    assert post_tool.request_type == "POST"
    assert post_tool.name == "addPet"

    schema_props = post_tool.input_schema["properties"]

    # Assert the $ref for Pet was resolved and inlined from in: body
    assert "name" in schema_props
    assert schema_props["name"]["type"] == "string"
    assert "photoUrls" in schema_props
    assert schema_props["photoUrls"]["type"] == "array"

    # Assert nested $ref for Category was resolved
    assert "category" in schema_props
    assert schema_props["category"]["type"] == "object"
    assert "name" in schema_props["category"]["properties"]

    # Assert required fields were migrated
    assert "name" in post_tool.input_schema["required"]
    assert "photoUrls" in post_tool.input_schema["required"]
