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
from cursor.thinking_split import CursorThinkingSplitter
from cursor.redacted_tools import RedactedToolStreamProcessor, extract_redacted_tool_calls
from cursor.bracket_tools import BracketToolCallProcessor, extract_bracket_tool_calls
from cursor.tokenizer import count_tokens, count_message_tokens
from cursor.config import FIRST_TOKEN_TIMEOUT, FIRST_TOKEN_MAX_RETRIES, SUPPRESS_THINKING_MODELS

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
    thinking_splitter = CursorThinkingSplitter()
    redacted_processor = RedactedToolStreamProcessor()
    bracket_processor = BracketToolCallProcessor()
    suppress_thinking = any(marker in model for marker in SUPPRESS_THINKING_MODELS)
    if suppress_thinking:
        logger.debug(f"Suppressing thinking blocks for model={model}")

    async def _emit_tool_use(tool: Dict[str, Any]) -> AsyncGenerator[str, None]:
        nonlocal current_block_index, thinking_block_started, thinking_block_index
        nonlocal text_block_started, text_block_index, tool_blocks

        if thinking_block_started and thinking_block_index is not None:
            yield _format_sse_event("content_block_stop", {
                "type": "content_block_stop", "index": thinking_block_index
            })
            thinking_block_started = False
            current_block_index += 1

        if text_block_started and text_block_index is not None:
            yield _format_sse_event("content_block_stop", {
                "type": "content_block_stop", "index": text_block_index
            })
            text_block_started = False
            current_block_index += 1

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
        yield _format_sse_event("content_block_stop", {
            "type": "content_block_stop", "index": current_block_index
        })

        tool_blocks.append({"id": tool_id, "name": tool_name, "input": tool_input})
        current_block_index += 1

    async def _emit_text_delta(text: str) -> AsyncGenerator[str, None]:
        nonlocal current_block_index, thinking_block_started, thinking_block_index
        nonlocal text_block_started, text_block_index, full_content

        if not text:
            return

        full_content += text

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

        yield _format_sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": text_block_index,
            "delta": {"type": "text_delta", "text": text}
        })

    async def _emit_thinking_delta(thinking: str) -> AsyncGenerator[str, None]:
        nonlocal thinking_block_started, thinking_block_index, current_block_index
        nonlocal full_thinking_content

        if not thinking:
            return

        full_thinking_content += thinking

        if not thinking_block_started:
            thinking_block_index = current_block_index
            yield _format_sse_event("content_block_start", {
                "type": "content_block_start",
                "index": thinking_block_index,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": f"sig_{uuid.uuid4().hex[:32]}",
                }
            })
            thinking_block_started = True

        yield _format_sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": thinking_block_index,
            "delta": {"type": "thinking_delta", "thinking": thinking}
        })

    async def _feed_visible_text(text: str) -> AsyncGenerator[str, None]:
        """Emit visible text after stripping embedded tool-call dialects.

        The composer-2.5 model occasionally narrates tool invocations using
        two different inline formats:

        * DeepSeek native tokens (``<｜tool▁call▁begin｜>...``) handled by
          ``RedactedToolStreamProcessor``.
        * Markdown-ish ``[Tool Call: Name({...})]`` handled by
          ``BracketToolCallProcessor``.

        We chain them so any tool call surfaced via either dialect is
        promoted to a structured ``tool_use`` block.
        """
        text_part, tools = redacted_processor.feed(text)
        text_part, bracket_tools = bracket_processor.feed(text_part)
        async for chunk in _emit_text_delta(text_part):
            yield chunk
        for tool in tools:
            async for chunk in _emit_tool_use(tool):
                yield chunk
        for tool in bracket_tools:
            async for chunk in _emit_tool_use(tool):
                yield chunk

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
                async for chunk in _feed_visible_text(event.content or ""):
                    yield chunk

            elif event.type == "thinking":
                raw_thinking = event.thinking_content or ""
                reasoning_part, visible_part = thinking_splitter.feed(raw_thinking)
                if not suppress_thinking:
                    async for chunk in _emit_thinking_delta(reasoning_part):
                        yield chunk
                else:
                    full_thinking_content += reasoning_part
                async for chunk in _feed_visible_text(visible_part):
                    yield chunk

            elif event.type == "tool_use" and event.tool_use:
                async for chunk in _emit_tool_use(event.tool_use):
                    yield chunk

            elif event.type == "error":
                error_data = event.error if event.error else "Unknown Cursor API error"
                logger.error(f"Cursor API error: {error_data}")
                yield _format_sse_event("error", {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"Cursor API error: {error_data}"
                    }
                })
                return  # Stop streaming on error

        flush_reasoning, flush_visible = thinking_splitter.flush()
        if not suppress_thinking:
            async for chunk in _emit_thinking_delta(flush_reasoning):
                yield chunk
        else:
            full_thinking_content += flush_reasoning
        async for chunk in _feed_visible_text(flush_visible):
            yield chunk

        flush_text, flush_tools = redacted_processor.flush()
        flush_text, flush_bracket_tools = bracket_processor.feed(flush_text)
        final_text, final_bracket_tools = bracket_processor.flush()
        flush_text += final_text
        flush_bracket_tools.extend(final_bracket_tools)
        async for chunk in _emit_text_delta(flush_text):
            yield chunk
        for tool in flush_tools:
            async for chunk in _emit_tool_use(tool):
                yield chunk
        for tool in flush_bracket_tools:
            async for chunk in _emit_tool_use(tool):
                yield chunk

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

    suppress_thinking = any(marker in model for marker in SUPPRESS_THINKING_MODELS)

    full_content = ""
    full_thinking = ""
    tool_calls = []
    thinking_splitter = CursorThinkingSplitter()

    try:
        async for event in parse_cursor_stream(response):
            if event.type == "content" and event.content:
                full_content += event.content
            elif event.type == "thinking" and event.thinking_content:
                reasoning_part, visible_part = thinking_splitter.feed(event.thinking_content)
                if not suppress_thinking:
                    full_thinking += reasoning_part
                full_content += visible_part
            elif event.type == "tool_use" and event.tool_use:
                tool_calls.append(event.tool_use)

        flush_reasoning, flush_visible = thinking_splitter.flush()
        if not suppress_thinking:
            full_thinking += flush_reasoning
        full_content += flush_visible

        full_content, redacted_tools = extract_redacted_tool_calls(full_content)
        tool_calls.extend(redacted_tools)
        full_content, bracket_tools = extract_bracket_tool_calls(full_content)
        tool_calls.extend(bracket_tools)
    finally:
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
