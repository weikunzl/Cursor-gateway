"""
Converters for transforming OpenAI format to Cursor format.

Adapter layer that converts OpenAI-specific formats
to the unified format used by converters_core.py.
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from cursor.models_openai import ChatMessage, ChatCompletionRequest, Tool
from cursor.converters_core import (
    extract_text_content,
    extract_images_from_content,
    UnifiedMessage,
    UnifiedTool,
    build_cursor_payload as core_build_cursor_payload,
)


def _extract_tool_results_from_openai(content: Any) -> List[Dict[str, Any]]:
    """Extracts tool results from OpenAI message content."""
    tool_results = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": item.get("tool_use_id", ""),
                    "content": extract_text_content(item.get("content", "")) or "(empty result)"
                })
    return tool_results


def _extract_tool_calls_from_openai(msg: ChatMessage) -> List[Dict[str, Any]]:
    """Extracts tool calls from OpenAI assistant message."""
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            if isinstance(tc, dict):
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}")
                    }
                })
    return tool_calls


def convert_openai_messages_to_unified(messages: List[ChatMessage]) -> Tuple[str, List[UnifiedMessage]]:
    """
    Converts OpenAI messages to unified format.

    Returns:
        Tuple of (system_prompt, unified_messages)
    """
    system_prompt = ""
    non_system_messages = []

    for msg in messages:
        if msg.role == "system":
            system_prompt += extract_text_content(msg.content) + "\n"
        else:
            non_system_messages.append(msg)

    system_prompt = system_prompt.strip()

    processed = []
    pending_tool_results = []

    for msg in non_system_messages:
        if msg.role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id or "",
                "content": extract_text_content(msg.content) or "(empty result)"
            }
            pending_tool_results.append(tool_result)
        else:
            if pending_tool_results:
                unified_msg = UnifiedMessage(
                    role="user",
                    content="",
                    tool_results=pending_tool_results.copy(),
                )
                processed.append(unified_msg)
                pending_tool_results.clear()

            tool_calls = None
            tool_results = None
            images = None

            if msg.role == "assistant":
                tool_calls = _extract_tool_calls_from_openai(msg) or None
            elif msg.role == "user":
                tool_results = _extract_tool_results_from_openai(msg.content) or None
                images = extract_images_from_content(msg.content) or None

            unified_msg = UnifiedMessage(
                role=msg.role,
                content=extract_text_content(msg.content),
                tool_calls=tool_calls,
                tool_results=tool_results,
                images=images,
            )
            processed.append(unified_msg)

    if pending_tool_results:
        unified_msg = UnifiedMessage(
            role="user",
            content="",
            tool_results=pending_tool_results.copy(),
        )
        processed.append(unified_msg)

    return system_prompt, processed


def convert_openai_tools_to_unified(tools: Optional[List[Tool]]) -> Optional[List[UnifiedTool]]:
    """Converts OpenAI tools to unified format."""
    if not tools:
        return None

    unified_tools = []
    for tool in tools:
        if tool.type != "function":
            continue

        if tool.function is not None:
            unified_tools.append(UnifiedTool(
                name=tool.function.name,
                description=tool.function.description,
                input_schema=tool.function.parameters,
            ))
        elif tool.name is not None:
            unified_tools.append(UnifiedTool(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            ))

    return unified_tools if unified_tools else None


def build_cursor_payload(
    request_data: ChatCompletionRequest,
    conversation_id: str,
) -> bytes:
    """
    Builds complete payload for Cursor API from OpenAI request.

    Returns:
        Protobuf-encoded bytes in ConnectRPC envelope
    """
    system_prompt, unified_messages = convert_openai_messages_to_unified(request_data.messages)
    unified_tools = convert_openai_tools_to_unified(request_data.tools)

    result = core_build_cursor_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=request_data.model,
        tools=unified_tools,
        conversation_id=conversation_id,
    )

    return result.payload
