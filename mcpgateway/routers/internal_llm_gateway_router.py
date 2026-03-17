# -*- coding: utf-8 -*-
"""Trusted internal endpoints used by the experimental Rust LLM Gateway."""

# Standard
import hmac

# Third-Party
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

# First-Party
from mcpgateway.config import settings
from mcpgateway.db import get_db
from mcpgateway.llm_schemas import ResolveChatCompletionTargetRequest
from mcpgateway.services.llm_provider_service import LLMModelNotFoundError, LLMProviderNotFoundError, LLMProviderService
from mcpgateway.services.llm_proxy_service import LLMProxyRequestError, LLMProxyService
from mcpgateway.services.logging_service import LoggingService

logger = LoggingService().get_logger(__name__)

internal_llm_gateway_router = APIRouter()
llm_provider_service = LLMProviderService()
llm_proxy_service = LLMProxyService()

_INTERNAL_FORWARD_HEADER = "x-forwarded-internally"
_INTERNAL_SECRET_HEADER = "x-contextforge-internal-secret"  # nosec B105 - header name, not a secret value


def _require_trusted_runtime_request(request: Request) -> None:
    """Allow only trusted loopback callers from the Rust sidecar.

    Args:
        request: Incoming FastAPI request.

    Raises:
        HTTPException: If the request is not from the trusted runtime path.
    """
    client_host = request.client.host if request.client else None
    from_loopback = client_host in ("127.0.0.1", "::1")
    is_internally_forwarded = request.headers.get(_INTERNAL_FORWARD_HEADER) == "true"
    if not from_loopback or not is_internally_forwarded:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Trusted runtime access required")

    expected_secret = settings.experimental_rust_llm_gateway_internal_secret
    if expected_secret:
        provided_secret = request.headers.get(_INTERNAL_SECRET_HEADER, "")
        if not hmac.compare_digest(provided_secret, expected_secret):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Trusted runtime access required")


@internal_llm_gateway_router.get(
    "/_internal/rust/llm/models",
    include_in_schema=False,
)
async def internal_list_models(
    request: Request,
    db: Session = Depends(get_db),
):
    """Return gateway-visible models for the Rust LLM Gateway.

    Args:
        request: Incoming trusted runtime request.
        db: Database session.

    Returns:
        An OpenAI-compatible model list payload.
    """
    _require_trusted_runtime_request(request)
    models = llm_provider_service.get_gateway_models(db)
    return {
        "object": "list",
        "data": [
            {
                "id": model.model_id,
                "object": "model",
                "created": 0,
                "owned_by": model.provider_name,
            }
            for model in models
        ],
    }


@internal_llm_gateway_router.post(
    "/_internal/rust/llm/resolve-chat-target",
    include_in_schema=False,
)
async def internal_resolve_chat_target(
    payload: ResolveChatCompletionTargetRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Resolve a model to trusted runtime metadata for the Rust sidecar.

    Args:
        payload: Model resolution request payload.
        request: Incoming trusted runtime request.
        db: Database session.

    Returns:
        Trusted runtime metadata for the requested model.

    Raises:
        HTTPException: If the request is untrusted or the target cannot be
            resolved.
    """
    _require_trusted_runtime_request(request)
    try:
        return llm_proxy_service.resolve_chat_completion_target(db, payload.model)
    except LLMModelNotFoundError as exc:
        logger.warning("Rust LLM runtime model not found: %s", payload.model)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except LLMProviderNotFoundError as exc:
        logger.warning("Rust LLM runtime provider not found for model: %s", payload.model)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except LLMProxyRequestError as exc:
        logger.error("Rust LLM runtime target resolution failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
