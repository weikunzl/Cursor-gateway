"""
Core converters for transforming unified format to Copilot OpenAI-compatible payload.

Defines the unified message format and builds Copilot API payloads.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


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
    """Result of building a Copilot payload."""
    payload: Dict[str, Any]  # OpenAI-format JSON dict
    model_id: str
    message_count: int


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

    Most LLM APIs expect alternating user/assistant messages.
    If we have consecutive messages with the same role, merge them.
    """
    if not messages:
        return messages

    merged = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"] and msg["role"] not in ("tool",):
            prev = merged[-1]
            prev_content = prev.get("content") or ""
            curr_content = msg.get("content") or ""
            prev["content"] = (prev_content + "\n\n" + curr_content).strip()
        else:
            merged.append(msg)

    return merged


def build_copilot_payload(
    messages: List[UnifiedMessage],
    system_prompt: str,
    model_id: str,
    tools: Optional[List[UnifiedTool]] = None,
    conversation_id: str = "",
) -> BuildResult:
    """
    Builds OpenAI-format JSON payload for Copilot's API.

    Args:
        messages: List of unified messages
        system_prompt: System prompt text
        model_id: Model ID for Copilot API
        tools: List of unified tools
        conversation_id: Conversation ID (unused, kept for interface parity)

    Returns:
        BuildResult with OpenAI-format dict payload
    """
    if not messages:
        raise ValueError("No messages to send")

    # Ensure last message is from user
    if messages[-1].role != "user":
        messages.append(UnifiedMessage(role="user", content="(empty)"))

    openai_messages: List[Dict[str, Any]] = []

    # System prompt as first message
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if msg.role == "assistant":
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or None,
            }
            if msg.tool_calls:
                formatted_tool_calls = []
                for tc in msg.tool_calls:
                    args = tc.get("function", {}).get("arguments", "{}")
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    formatted_tool_calls.append({
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": args,
                        },
                    })
                assistant_msg["tool_calls"] = formatted_tool_calls
            openai_messages.append(assistant_msg)

        elif msg.role == "user":
            # Tool results become separate tool messages
            if msg.tool_results:
                for tr in msg.tool_results:
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr.get("content", "(empty result)"),
                    })
                # Only add user message if there's actual text content
                if msg.content:
                    openai_messages.append({"role": "user", "content": msg.content})
            else:
                content: Any = msg.content or "(empty)"
                # Re-attach images as content blocks if present
                if msg.images:
                    content_blocks: List[Dict[str, Any]] = []
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})
                    for img in msg.images:
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{img['media_type']};base64,{img['data']}"
                            },
                        })
                    content = content_blocks
                openai_messages.append({"role": "user", "content": content})
        else:
            openai_messages.append({"role": msg.role, "content": msg.content or "(empty)"})

    # Merge consecutive same-role messages (excluding tool messages)
    openai_messages = _merge_consecutive_messages(openai_messages)

    # Build tools array in OpenAI format
    openai_tools: List[Dict[str, Any]] = []
    if tools:
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema or {},
                },
            })

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": openai_messages,
        "stream": True,
    }
    if openai_tools:
        payload["tools"] = openai_tools

    logger.debug(
        f"Built Copilot payload: model={model_id}, messages={len(openai_messages)}, "
        f"system_prompt_len={len(system_prompt)}, tools={len(openai_tools)}"
    )

    return BuildResult(
        payload=payload,
        model_id=model_id,
        message_count=len(openai_messages),
    )
