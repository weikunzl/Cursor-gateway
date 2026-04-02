"""
Pydantic models for Anthropic Messages API.

Defines data schemas for requests and responses compatible with
Anthropic's Messages API specification.

Reference: https://docs.anthropic.com/en/api/messages
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ==================================================================================================
# Content Block Models
# ==================================================================================================

class TextContentBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingContentBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class ToolUseContentBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultContentBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[
        Union[str, List[Union["TextContentBlock", "ImageContentBlock"]]]
    ] = None
    is_error: Optional[bool] = None


# ==================================================================================================
# Image Content Block Models
# ==================================================================================================

class Base64ImageSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class URLImageSource(BaseModel):
    type: Literal["url"] = "url"
    url: str


class ImageContentBlock(BaseModel):
    type: Literal["image"] = "image"
    source: Union[Base64ImageSource, URLImageSource]


ContentBlock = Union[
    TextContentBlock,
    ThinkingContentBlock,
    ImageContentBlock,
    ToolUseContentBlock,
    ToolResultContentBlock,
]


# ==================================================================================================
# Message Models
# ==================================================================================================

class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]

    model_config = {"extra": "allow"}


# ==================================================================================================
# Tool Models
# ==================================================================================================

class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ToolChoiceAuto(BaseModel):
    type: Literal["auto"] = "auto"


class ToolChoiceAny(BaseModel):
    type: Literal["any"] = "any"


class ToolChoiceTool(BaseModel):
    type: Literal["tool"] = "tool"
    name: str


ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool]


# ==================================================================================================
# Request Models
# ==================================================================================================

class SystemContentBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    cache_control: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


SystemPrompt = Union[str, List[SystemContentBlock], List[Dict[str, Any]]]


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage] = Field(min_length=1)
    max_tokens: int

    system: Optional[SystemPrompt] = None
    stream: bool = False

    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Union[ToolChoice, Dict[str, Any]]] = None

    temperature: Optional[float] = Field(default=None, ge=0, le=1)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)

    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Response Models
# ==================================================================================================

class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Union[ThinkingContentBlock, TextContentBlock, ToolUseContentBlock]]
    model: str
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# ==================================================================================================
# Streaming Event Models
# ==================================================================================================

class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: Dict[str, Any]


class ContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: Dict[str, Any]


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class InputJsonDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Union[TextDelta, ThinkingDelta, InputJsonDelta, Dict[str, Any]]


class ContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaUsage(BaseModel):
    output_tokens: int


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any]
    usage: MessageDeltaUsage


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: Dict[str, Any]


StreamingEvent = Union[
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent,
    ErrorEvent,
]


# ==================================================================================================
# Error Models
# ==================================================================================================

class AnthropicErrorDetail(BaseModel):
    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicErrorDetail
