"""
Converters for transforming Anthropic Messages API format to Cursor format.

Adapter layer that converts Anthropic-specific formats
to the unified format used by converters_core.py.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from cursor.models_anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessage,
    AnthropicTool,
)
from cursor.converters_core import (
    BuildResult,
    UnifiedMessage,
    UnifiedTool,
    build_cursor_payload as core_build_cursor_payload,
    extract_text_content,
    extract_images_from_content,
)


def convert_anthropic_content_to_text(content: Any) -> str:
    """Extracts text content from Anthropic message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)
        return "".join(text_parts)
    return str(content) if content else ""


def extract_system_prompt(system: Any) -> str:
    """Extracts system prompt from Anthropic system field."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        text_parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif hasattr(block, "type") and block.type == "text":
                text_parts.append(getattr(block, "text", ""))
        return "\n".join(text_parts)
    return str(system)


def extract_tool_results_from_anthropic_content(content: Any) -> List[Dict[str, Any]]:
    """Extracts tool results from Anthropic message content."""
    tool_results = []
    if not isinstance(content, list):
        return tool_results

    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        tool_use_id = block.get("tool_use_id") if isinstance(block, dict) else getattr(block, "tool_use_id", None)
        result_content = block.get("content", "") if isinstance(block, dict) else getattr(block, "content", "")

        if block_type == "tool_result" and tool_use_id:
            if isinstance(result_content, list):
                result_content = extract_text_content(result_content)
            elif not isinstance(result_content, str):
                result_content = str(result_content) if result_content else ""

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_content or "(empty result)",
            })

    return tool_results


def extract_tool_uses_from_anthropic_content(content: Any) -> List[Dict[str, Any]]:
    """Extracts tool uses from Anthropic assistant message content."""
    tool_calls = []
    if not isinstance(content, list):
        return tool_calls

    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            tool_id = block.get("id")
            tool_name = block.get("name")
            tool_input = block.get("input", {})
        elif hasattr(block, "type"):
            block_type = block.type
            tool_id = getattr(block, "id", None)
            tool_name = getattr(block, "name", None)
            tool_input = getattr(block, "input", {})
        else:
            continue

        if block_type == "tool_use" and tool_id and tool_name:
            tool_calls.append({
                "id": tool_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_input if isinstance(tool_input, str) else tool_input,
                },
            })

    return tool_calls


def convert_anthropic_messages(messages: List[AnthropicMessage]) -> List[UnifiedMessage]:
    """Converts Anthropic messages to unified format."""
    unified_messages = []

    for msg in messages:
        role = msg.role
        content = msg.content
        text_content = convert_anthropic_content_to_text(content)

        tool_calls = None
        tool_results = None
        images = None

        if role == "assistant":
            tool_calls = extract_tool_uses_from_anthropic_content(content) or None
        elif role == "user":
            tool_results = extract_tool_results_from_anthropic_content(content) or None
            images = extract_images_from_content(content) or None

        unified_messages.append(UnifiedMessage(
            role=role,
            content=text_content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            images=images,
        ))

    return unified_messages


def convert_anthropic_tools(tools: Optional[List[AnthropicTool]]) -> Optional[List[UnifiedTool]]:
    """Converts Anthropic tools to unified format."""
    if not tools:
        return None

    unified_tools = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description")
            input_schema = tool.get("input_schema", {})
        else:
            name = tool.name
            description = tool.description
            input_schema = tool.input_schema

        unified_tools.append(UnifiedTool(name=name, description=description, input_schema=input_schema))

    return unified_tools if unified_tools else None


def anthropic_to_cursor(
    request: AnthropicMessagesRequest,
    conversation_id: str,
) -> "BuildResult":
    """
    Convert an Anthropic Messages API request into a Cursor payload.

    Args:
        request: Validated Anthropic Messages API request body.
        conversation_id: Stable conversation identifier (sent to Cursor).

    Returns:
        ``BuildResult`` carrying the protobuf-encoded ConnectRPC envelope
        together with the ``compressed`` flag the HTTP layer needs to
        advertise via ``Connect-Content-Encoding``.
    """
    unified_messages = convert_anthropic_messages(request.messages)
    unified_tools = convert_anthropic_tools(request.tools)
    system_prompt = extract_system_prompt(request.system)

    return core_build_cursor_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=request.model,
        tools=unified_tools,
        conversation_id=conversation_id,
    )
