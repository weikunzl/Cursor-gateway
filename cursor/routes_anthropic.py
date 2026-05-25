"""
FastAPI routes for Anthropic Messages API.

Endpoint: /v1/messages
"""

import json
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Security, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from cursor.config import PROXY_API_KEY, CURSOR_API_HOST, CURSOR_CHAT_RPC
from cursor.models_anthropic import AnthropicMessagesRequest
from cursor.auth import CursorAuthManager
from cursor.cache import ModelInfoCache
from cursor.model_resolver import ModelResolver
from cursor.converters_anthropic import anthropic_to_cursor
from cursor.streaming_anthropic import stream_cursor_to_anthropic, collect_anthropic_response
from cursor.http_client import CursorHttpClient
from cursor.utils import generate_conversation_id


# --- Security ---
anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
auth_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_anthropic_api_key(
    x_api_key: Optional[str] = Security(anthropic_api_key_header),
    authorization: Optional[str] = Security(auth_header),
) -> bool:
    if x_api_key and x_api_key == PROXY_API_KEY:
        return True
    if authorization and authorization == f"Bearer {PROXY_API_KEY}":
        return True
    raise HTTPException(
        status_code=401,
        detail={"type": "error", "error": {"type": "authentication_error", "message": "Invalid or missing API key."}}
    )


router = APIRouter(tags=["Anthropic API"])


@router.post("/v1/messages", dependencies=[Depends(verify_anthropic_api_key)])
async def messages(
    request: Request,
    request_data: AnthropicMessagesRequest,
    anthropic_version: Optional[str] = Header(None, alias="anthropic-version"),
):
    logger.info(f"Request to /v1/messages (model={request_data.model}, stream={request_data.stream})")

    auth_manager: CursorAuthManager = request.app.state.auth_manager
    model_cache: ModelInfoCache = request.app.state.model_cache
    model_resolver: ModelResolver = request.app.state.model_resolver

    # Resolve model
    resolved_model = model_resolver.resolve(request_data.model)
    request_data.model = resolved_model

    conversation_id = generate_conversation_id()

    try:
        cursor_payload = anthropic_to_cursor(request_data, conversation_id)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}}
        )

    url = f"{CURSOR_API_HOST}{CURSOR_CHAT_RPC}"

    if request_data.stream:
        http_client = CursorHttpClient(auth_manager, shared_client=None)
    else:
        shared_client = request.app.state.http_client
        http_client = CursorHttpClient(auth_manager, shared_client=shared_client)

    messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]

    try:
        response = await http_client.request_with_retry("POST", url, cursor_payload, stream=True)

        if response.status_code != 200:
            try:
                error_content = await response.aread()
            except Exception:
                error_content = b"Unknown error"
            await http_client.close()
            error_text = error_content.decode("utf-8", errors="replace")
            logger.warning(f"HTTP {response.status_code} - POST /v1/messages - {error_text[:200]}")
            return JSONResponse(
                status_code=response.status_code,
                content={"type": "error", "error": {"type": "api_error", "message": error_text}}
            )

        if request_data.stream:
            async def stream_wrapper():
                try:
                    async for chunk in stream_cursor_to_anthropic(
                        response, request_data.model, model_cache,
                        request_messages=messages_for_tokenizer,
                    ):
                        yield chunk
                except GeneratorExit:
                    logger.debug("Client disconnected during streaming")
                except Exception as e:
                    try:
                        yield f'event: error\ndata: {json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e)}})}\n\n'
                    except Exception:
                        pass
                finally:
                    await http_client.close()

            return StreamingResponse(
                stream_wrapper(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            anthropic_response = await collect_anthropic_response(
                response, request_data.model, model_cache,
                request_messages=messages_for_tokenizer,
            )
            await http_client.close()
            return JSONResponse(content=anthropic_response)

    except HTTPException:
        await http_client.close()
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"type": "error", "error": {"type": "api_error", "message": f"Internal Server Error: {e}"}}
        )
