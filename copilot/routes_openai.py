"""
FastAPI routes for OpenAI-compatible API.

Endpoints:
- / and /health: Health check
- /v1/models: Models list
- /v1/chat/completions: Chat completions
"""

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from copilot.config import PROXY_API_KEY, APP_VERSION, COPILOT_CHAT_ENDPOINT
from copilot.models_openai import OpenAIModel, ModelList, ChatCompletionRequest
from copilot.auth import CopilotAuthManager
from copilot.cache import ModelInfoCache
from copilot.model_resolver import ModelResolver
from copilot.converters_openai import build_copilot_payload
from copilot.streaming_openai import stream_copilot_to_openai, collect_stream_response
from copilot.http_client import CopilotHttpClient
from copilot.utils import generate_conversation_id


# --- Security ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(auth_header: str = Depends(api_key_header)) -> bool:
    if not auth_header or auth_header != f"Bearer {PROXY_API_KEY}":
        logger.warning("Access attempt with invalid API key.")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


router = APIRouter()


@router.get("/")
async def root():
    return {"status": "ok", "message": "Copilot Gateway is running", "version": APP_VERSION}


@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "version": APP_VERSION}


@router.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
async def get_models(request: Request):
    logger.info("Request to /v1/models")
    model_resolver: ModelResolver = request.app.state.model_resolver
    available_model_ids = model_resolver.get_available_models()

    openai_models = [
        OpenAIModel(id=model_id, owned_by="github", description="Model via GitHub Copilot API")
        for model_id in available_model_ids
    ]
    return ModelList(data=openai_models)


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
    logger.info(f"Request to /v1/chat/completions (model={request_data.model}, stream={request_data.stream})")

    auth_manager: CopilotAuthManager = request.app.state.auth_manager
    model_cache: ModelInfoCache = request.app.state.model_cache
    model_resolver: ModelResolver = request.app.state.model_resolver

    resolved_model = model_resolver.resolve(request_data.model)
    request_data.model = resolved_model

    conversation_id = generate_conversation_id()

    try:
        copilot_payload = build_copilot_payload(request_data, conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if request_data.stream:
        http_client = CopilotHttpClient(auth_manager, shared_client=None)
    else:
        shared_client = request.app.state.http_client
        http_client = CopilotHttpClient(auth_manager, shared_client=shared_client)

    messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
    tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None

    try:
        url = auth_manager.api_host.rstrip("/") + COPILOT_CHAT_ENDPOINT
        response = await http_client.request_with_retry("POST", url, copilot_payload, stream=True)

        if response.status_code != 200:
            try:
                error_content = await response.aread()
            except Exception:
                error_content = b"Unknown error"
            await http_client.close()
            error_text = error_content.decode("utf-8", errors="replace")
            logger.warning(f"HTTP {response.status_code} - POST /v1/chat/completions - {error_text[:200]}")
            return JSONResponse(
                status_code=response.status_code,
                content={"error": {"message": error_text, "type": "copilot_api_error", "code": response.status_code}}
            )

        if request_data.stream:
            async def stream_wrapper():
                try:
                    async for chunk in stream_copilot_to_openai(
                        response, request_data.model, model_cache,
                        request_messages=messages_for_tokenizer,
                        request_tools=tools_for_tokenizer,
                    ):
                        yield chunk
                except GeneratorExit:
                    logger.debug("Client disconnected during streaming")
                except Exception as e:
                    try:
                        yield "data: [DONE]\n\n"
                    except Exception:
                        pass
                    raise
                finally:
                    await http_client.close()

            return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
        else:
            openai_response = await collect_stream_response(
                response, request_data.model, model_cache,
                request_messages=messages_for_tokenizer,
                request_tools=tools_for_tokenizer,
            )
            await http_client.close()
            return JSONResponse(content=openai_response)

    except HTTPException:
        await http_client.close()
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
