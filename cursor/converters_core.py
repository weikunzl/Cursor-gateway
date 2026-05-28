"""
Core converters for transforming unified format to Cursor protobuf payload.

Defines the unified message format and builds Cursor API payloads.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

import platform

from cursor.protobuf import encode_chat_request, wrap_connect_envelope
from cursor.utils import generate_message_id, _platform_os, _platform_arch

def _platform_os_version() -> str:
    if platform.system() == "Darwin":
        return platform.mac_ver()[0] or platform.release()
    return platform.release()


@dataclass
class UnifiedMessage:
    """Unified message format used across all converters."""
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None


@dataclass
class UnifiedTool:
    """Unified tool format."""
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


@dataclass
class BuildResult:
    """Result of building a Cursor payload.

    Attributes:
        payload: Protobuf-encoded bytes wrapped in a ConnectRPC envelope.
        model_id: Resolved model identifier sent to Cursor.
        message_count: Number of protobuf messages in the request.
        compressed: True if ``payload`` is a gzip-compressed envelope (first
            flag byte = ``0x01``). Callers must advertise this to the server
            via the ``Connect-Content-Encoding`` request header — otherwise
            Cursor's backend rejects the call with ``"received compressed
            envelope, but do not know how to decompress"``.
    """
    payload: bytes
    model_id: str
    message_count: int
    compressed: bool = False


def extract_text_content(content: Any) -> str:
    """
    Extracts text from various content formats.

    Handles:
    - String content
    - List of content blocks [{"type": "text", "text": "..."}]
    - None
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def extract_images_from_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts images from content blocks.

    Handles OpenAI image_url format and Anthropic image format.
    """
    images = []
    if not isinstance(content, list):
        return images

    for item in content:
        if not isinstance(item, dict):
            continue

        # OpenAI format: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        if item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                # Parse data URI
                try:
                    header, data = url.split(",", 1)
                    media_type = header.split(";")[0].split(":")[1]
                    images.append({"media_type": media_type, "data": data})
                except (ValueError, IndexError):
                    pass

        # Anthropic format: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        elif item.get("type") == "image":
            source = item.get("source", {})
            if isinstance(source, dict) and source.get("type") == "base64":
                images.append({
                    "media_type": source.get("media_type", "image/png"),
                    "data": source.get("data", "")
                })

    return images


def _merge_consecutive_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive messages with the same role.

    Cursor API (like most LLM APIs) expects alternating user/assistant messages.
    If we have consecutive messages with the same role, merge them.
    """
    if not messages:
        return messages

    merged = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            # Merge content
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)

    return merged


def build_cursor_payload(
    messages: List[UnifiedMessage],
    system_prompt: str,
    model_id: str,
    tools: Optional[List[UnifiedTool]] = None,
    conversation_id: str = "",
) -> BuildResult:
    """
    Builds protobuf-encoded payload for Cursor's ConnectRPC API.

    Args:
        messages: List of unified messages
        system_prompt: System prompt text
        model_id: Model ID for Cursor API
        tools: List of unified tools (currently appended to system prompt)
        conversation_id: Conversation ID

    Returns:
        BuildResult with protobuf bytes
    """
    if not messages:
        raise ValueError("No messages to send")

    # Ensure last message is from user (required by most LLM APIs)
    # If last message is not user, add a synthetic one
    if messages[-1].role != "user":
        messages.append(UnifiedMessage(role="user", content="(empty)"))

    # Build system prompt with tool descriptions if tools are provided
    full_system_prompt = system_prompt
    if tools:
        tool_descriptions = []
        for tool in tools:
            desc = f"## Tool: {tool.name}\n"
            if tool.description:
                desc += f"{tool.description}\n"
            if tool.input_schema:
                import json
                desc += f"Parameters: {json.dumps(tool.input_schema, ensure_ascii=False)}\n"
            tool_descriptions.append(desc)

        if tool_descriptions:
            tools_section = "\n\n# Available Tools\n\n" + "\n".join(tool_descriptions)
            full_system_prompt = (full_system_prompt + tools_section) if full_system_prompt else tools_section.strip()

    # Convert unified messages to simple dicts for protobuf encoding
    proto_messages = []
    for msg in messages:
        content = msg.content

        # Append tool results to content
        if msg.tool_results:
            tool_result_text = ""
            for tr in msg.tool_results:
                tool_result_text += f"\n[Tool Result for {tr.get('tool_use_id', 'unknown')}]: {tr.get('content', '')}"
            content = (content + tool_result_text) if content else tool_result_text.strip()

        # Append tool calls to content (for assistant messages)
        if msg.tool_calls:
            import json
            tool_call_text = ""
            for tc in msg.tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                tool_call_text += f"\n[Tool Call: {name}({args})]"
            content = (content + tool_call_text) if content else tool_call_text.strip()

        if not content:
            content = "(empty)"

        proto_messages.append({
            "role": msg.role,
            "content": content,
            "message_id": generate_message_id(),
        })

    # Merge consecutive same-role messages
    proto_messages = _merge_consecutive_messages(proto_messages)

    # Determine if agent mode (when tools are provided)
    use_agent = bool(tools)
    logger.info(f"Building Cursor payload: is_agentic={use_agent}, tools={len(tools) if tools else 0}")

    # Encode to protobuf
    proto_bytes = encode_chat_request(
        messages=proto_messages,
        model=model_id,
        system_prompt=full_system_prompt,
        conversation_id=conversation_id,
        client_os=_platform_os(),
        client_arch=_platform_arch(),
        client_os_version=_platform_os_version(),
        is_agentic=use_agent,
    )

    # Wrap in ConnectRPC envelope
    # Compress if payload is large (>= 3 messages as per Cursor's behavior)
    compress = len(proto_messages) >= 3
    envelope = wrap_connect_envelope(proto_bytes, compress=compress)

    logger.debug(
        f"Built Cursor payload: model={model_id}, messages={len(proto_messages)}, "
        f"system_prompt_len={len(full_system_prompt)}, compressed={compress}, "
        f"payload_size={len(envelope)} bytes"
    )

    return BuildResult(
        payload=envelope,
        model_id=model_id,
        message_count=len(proto_messages),
        compressed=compress,
    )
