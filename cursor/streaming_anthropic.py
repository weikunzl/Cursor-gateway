"""
Streaming logic for converting Cursor stream to Anthropic Messages API format.

Formats CursorEvent objects into Anthropic SSE events.
"""

import json
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Optional, Any

import httpx
from loguru import logger

from cursor.streaming_core import (
    parse_cursor_stream,
    FirstTokenTimeoutError,
    stream_with_first_token_retry,
)
from cursor.tokenizer import count_tokens, count_message_tokens
from cursor.config import FIRST_TOKEN_TIMEOUT, FIRST_TOKEN_MAX_RETRIES

if TYPE_CHECKING:
    from cursor.auth import CursorAuthManager
    from cursor.cache import ModelInfoCache


def _generate_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_cursor_to_anthropic(
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
) -> AsyncGenerator[str, None]:
    """
    Generator for converting Cursor stream to Anthropic SSE format.

    Yields:
        Strings in Anthropic SSE format
    """
    message_id = _generate_message_id()
    input_tokens = 0
    output_tokens = 0
    full_content = ""
    full_thinking_content = ""

    if request_messages:
        input_tokens = count_message_tokens(request_messages, apply_claude_correction=False)

    current_block_index = 0
    thinking_block_started = False
    thinking_block_index: Optional[int] = None
    text_block_started = False
    text_block_index: Optional[int] = None
    tool_blocks: List[Dict[str, Any]] = []

    try:
        # Send message_start
        yield _format_sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        })

        async for event in parse_cursor_stream(response, first_token_timeout):
            if event.type == "content":
                content = event.content or ""
                full_content += content

                # Close thinking block if transitioning to content
                if thinking_block_started and thinking_block_index is not None:
                    yield _format_sse_event("content_block_stop", {
                        "type": "content_block_stop", "index": thinking_block_index
                    })
                    thinking_block_started = False
                    current_block_index += 1

                if not text_block_started:
                    text_block_index = current_block_index
                    yield _format_sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": text_block_index,
                        "content_block": {"type": "text", "text": ""}
                    })
                    text_block_started = True

                if content:
                    yield _format_sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": content}
                    })

            elif event.type == "thinking":
                thinking = event.thinking_content or ""
                full_thinking_content += thinking

                if not thinking_block_started:
                    thinking_block_index = current_block_index
                    yield _format_sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": thinking_block_index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": f"sig_{uuid.uuid4().hex[:32]}"}
                    })
                    thinking_block_started = True

                if thinking:
                    yield _format_sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": thinking_block_index,
                        "delta": {"type": "thinking_delta", "thinking": thinking}
                    })

            elif event.type == "tool_use" and event.tool_use:
                # Close open blocks
                if thinking_block_started and thinking_block_index is not None:
                    yield _format_sse_event("content_block_stop", {"type": "content_block_stop", "index": thinking_block_index})
                    thinking_block_started = False
                    current_block_index += 1
                if text_block_started and text_block_index is not None:
                    yield _format_sse_event("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                    text_block_started = False
                    current_block_index += 1

                tool = event.tool_use
                tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                tool_name = tool.get("name", "")
                tool_input = tool.get("arguments", {})
                if isinstance(tool_input, str):
                    try:
                        tool_input = json.loads(tool_input)
                    except json.JSONDecodeError:
                        tool_input = {}

                yield _format_sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": current_block_index,
                    "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}}
                })
                yield _format_sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": current_block_index,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input, ensure_ascii=False)}
                })
                yield _format_sse_event("content_block_stop", {"type": "content_block_stop", "index": current_block_index})

                tool_blocks.append({"id": tool_id, "name": tool_name, "input": tool_input})
                current_block_index += 1

        # Close remaining blocks
        if thinking_block_started and thinking_block_index is not None:
            yield _format_sse_event("content_block_stop", {"type": "content_block_stop", "index": thinking_block_index})
        if text_block_started and text_block_index is not None:
            yield _format_sse_event("content_block_stop", {"type": "content_block_stop", "index": text_block_index})

        output_tokens = count_tokens(full_content + full_thinking_content)
        stop_reason = "tool_use" if tool_blocks else "end_turn"

        yield _format_sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens}
        })
        yield _format_sse_event("message_stop", {"type": "message_stop"})

    except FirstTokenTimeoutError:
        raise
    except GeneratorExit:
        logger.debug("Client disconnected (GeneratorExit)")
        raise
    except Exception as e:
        logger.error(f"Error during Anthropic streaming: {type(e).__name__}: {e}", exc_info=True)
        yield _format_sse_event("error", {
            "type": "error",
            "error": {"type": "api_error", "message": f"Internal error: {e}"}
        })
        raise
    finally:
        try:
            await response.aclose()
        except Exception:
            pass


async def collect_anthropic_response(
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    request_messages: Optional[list] = None,
) -> dict:
    """Collect full response in Anthropic format for non-streaming mode."""
    message_id = _generate_message_id()
    input_tokens = 0
    if request_messages:
        input_tokens = count_message_tokens(request_messages, apply_claude_correction=False)

    full_content = ""
    full_thinking = ""
    tool_calls = []

    async for event in parse_cursor_stream(response):
        if event.type == "content" and event.content:
            full_content += event.content
        elif event.type == "thinking" and event.thinking_content:
            full_thinking += event.thinking_content
        elif event.type == "tool_use" and event.tool_use:
            tool_calls.append(event.tool_use)

    try:
        await response.aclose()
    except Exception:
        pass

    content_blocks = []
    if full_thinking:
        content_blocks.append({
            "type": "thinking",
            "thinking": full_thinking,
            "signature": f"sig_{uuid.uuid4().hex[:32]}"
        })
    if full_content:
        content_blocks.append({"type": "text", "text": full_content})

    for tc in tool_calls:
        tool_input = tc.get("arguments", {})
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                tool_input = {}
        content_blocks.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": tc.get("name", ""),
            "input": tool_input,
        })

    output_tokens = count_tokens(full_content + full_thinking)
    stop_reason = "tool_use" if tool_calls else "end_turn"

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
