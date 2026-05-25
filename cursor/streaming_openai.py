"""
Streaming logic for converting Cursor stream to OpenAI format.

Converts CursorEvent objects to OpenAI chat.completion.chunk SSE format.
"""

import json
import time
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Awaitable, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from cursor.utils import generate_completion_id
from cursor.config import FIRST_TOKEN_TIMEOUT, FIRST_TOKEN_MAX_RETRIES
from cursor.tokenizer import count_tokens, count_message_tokens, count_tools_tokens
from cursor.streaming_core import (
    parse_cursor_stream,
    FirstTokenTimeoutError,
    CursorEvent,
    stream_with_first_token_retry as stream_with_first_token_retry_core,
)

if TYPE_CHECKING:
    from cursor.auth import CursorAuthManager
    from cursor.cache import ModelInfoCache


async def stream_cursor_to_openai_internal(
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """
    Internal generator for converting Cursor stream to OpenAI format.

    Yields:
        Strings in SSE format: "data: {...}\\n\\n" or "data: [DONE]\\n\\n"

    Raises:
        FirstTokenTimeoutError: If first token not received within timeout
    """
    completion_id = generate_completion_id()
    created_time = int(time.time())
    first_chunk = True

    full_content = ""
    full_thinking_content = ""
    tool_calls_from_stream = []

    try:
        async for event in parse_cursor_stream(response, first_token_timeout):
            if event.type == "content" and event.content:
                full_content += event.content

                delta = {"content": event.content}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False

                openai_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                }
                yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"

            elif event.type == "thinking" and event.thinking_content:
                full_thinking_content += event.thinking_content

                delta = {"reasoning_content": event.thinking_content}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False

                openai_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                }
                yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"

            elif event.type == "tool_use" and event.tool_use:
                tool_calls_from_stream.append(event.tool_use)

            elif event.type == "usage" and event.usage:
                pass  # Usage handled at the end

            elif event.type == "error" and event.error:
                error_text = str(event.error) if not isinstance(event.error, dict) else json.dumps(event.error, ensure_ascii=False)
                logger.warning(f"Cursor upstream error: {error_text}")
                raise HTTPException(status_code=502, detail=f"Cursor API error: {error_text}")

        # Determine finish_reason
        finish_reason = "tool_calls" if tool_calls_from_stream else "stop"

        # Count tokens
        completion_tokens = count_tokens(full_content + full_thinking_content)
        prompt_tokens = 0
        if request_messages:
            prompt_tokens = count_message_tokens(request_messages, apply_claude_correction=False)
            if request_tools:
                prompt_tokens += count_tools_tokens(request_tools, apply_claude_correction=False)
        total_tokens = prompt_tokens + completion_tokens

        # Send tool calls if present
        if tool_calls_from_stream:
            from cursor.utils import generate_tool_call_id
            indexed_tool_calls = []
            for idx, tc in enumerate(tool_calls_from_stream):
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", "{}")
                indexed_tc = {
                    "index": idx,
                    "id": generate_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args if isinstance(tool_args, str) else json.dumps(tool_args, ensure_ascii=False),
                    }
                }
                indexed_tool_calls.append(indexed_tc)

            tool_calls_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": indexed_tool_calls},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(tool_calls_chunk, ensure_ascii=False)}\n\n"

        # Final chunk with usage
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except FirstTokenTimeoutError:
        raise
    except GeneratorExit:
        logger.debug("Client disconnected (GeneratorExit)")
    except Exception as e:
        logger.error(f"Error during streaming: {type(e).__name__}: {e}", exc_info=True)
        raise
    finally:
        try:
            await response.aclose()
        except Exception:
            pass


async def stream_cursor_to_openai(
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """Wrapper generator without retry logic."""
    async for chunk in stream_cursor_to_openai_internal(
        response, model, model_cache,
        request_messages=request_messages,
        request_tools=request_tools,
    ):
        yield chunk


async def stream_with_first_token_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    model: str,
    model_cache: "ModelInfoCache",
    max_retries: int = FIRST_TOKEN_MAX_RETRIES,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """Streaming with automatic retry on first token timeout."""

    def create_http_error(status_code: int, error_text: str) -> HTTPException:
        return HTTPException(status_code=status_code, detail=f"Upstream API error: {error_text}")

    def create_timeout_error(retries: int, timeout: float) -> HTTPException:
        return HTTPException(status_code=504, detail=f"Model did not respond within {timeout}s after {retries} attempts.")

    async def stream_processor(response: httpx.Response) -> AsyncGenerator[str, None]:
        async for chunk in stream_cursor_to_openai_internal(
            response, model, model_cache,
            first_token_timeout=first_token_timeout,
            request_messages=request_messages,
            request_tools=request_tools,
        ):
            yield chunk

    async for chunk in stream_with_first_token_retry_core(
        make_request=make_request,
        stream_processor=stream_processor,
        max_retries=max_retries,
        first_token_timeout=first_token_timeout,
        on_http_error=create_http_error,
        on_all_retries_failed=create_timeout_error,
    ):
        yield chunk


async def collect_stream_response(
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> dict:
    """Collect full response from streaming for non-streaming mode."""
    full_content = ""
    full_reasoning_content = ""
    final_usage = None
    tool_calls = []
    completion_id = generate_completion_id()

    async for chunk_str in stream_cursor_to_openai(
        response, model, model_cache,
        request_messages=request_messages,
        request_tools=request_tools,
    ):
        if not chunk_str.startswith("data:"):
            continue
        data_str = chunk_str[len("data:"):].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            chunk_data = json.loads(data_str)
            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                full_content += delta["content"]
            if "reasoning_content" in delta:
                full_reasoning_content += delta["reasoning_content"]
            if "tool_calls" in delta:
                tool_calls.extend(delta["tool_calls"])
            if "usage" in chunk_data:
                final_usage = chunk_data["usage"]
        except (json.JSONDecodeError, IndexError):
            continue

    message = {"role": "assistant", "content": full_content}
    if full_reasoning_content:
        message["reasoning_content"] = full_reasoning_content
    if tool_calls:
        cleaned = []
        for tc in tool_calls:
            func = tc.get("function") or {}
            cleaned.append({
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {"name": func.get("name", ""), "arguments": func.get("arguments", "{}")}
            })
        message["tool_calls"] = cleaned

    usage = final_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": usage,
    }
