"""
Pydantic models for OpenAI-compatible API.

Defines data schemas for requests and responses,
providing validation and serialization.
"""

import time
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field


# ==================================================================================================
# Models for /v1/models endpoint
# ==================================================================================================

class OpenAIModel(BaseModel):
    """Data model for describing an AI model in OpenAI format."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "github"
    description: Optional[str] = None


class ModelList(BaseModel):
    """List of models in OpenAI format. Response of GET /v1/models endpoint."""
    object: str = "list"
    data: List[OpenAIModel]


# ==================================================================================================
# Models for /v1/chat/completions endpoint
# ==================================================================================================

class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""
    role: str
    content: Optional[Union[str, List[Any], Any]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None

    model_config = {"extra": "allow"}


class ToolFunction(BaseModel):
    """Tool function description."""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """Tool in OpenAI format."""
    type: str = "function"
    function: Optional[ToolFunction] = None
    name: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    """Request for response generation in OpenAI Chat Completions API format."""
    model: str
    messages: Annotated[List[ChatMessage], Field(min_length=1)]
    stream: bool = False

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict]] = None

    stream_options: Optional[Dict[str, Any]] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Models for responses
# ==================================================================================================

class ChatCompletionChoice(BaseModel):
    """Single response variant in Chat Completion."""
    index: int = 0
    message: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Full Chat Completion response (non-streaming)."""
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(BaseModel):
    """Delta of changes in streaming chunk."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChunkChoice(BaseModel):
    """Single variant in streaming chunk."""
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk in OpenAI format."""
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[ChatCompletionUsage] = None
