"""
Cursor Gateway - Proxy for Cursor API.

This package provides a modular architecture for proxying
OpenAI and Anthropic API requests to Cursor's ConnectRPC backend.

Modules:
    - config: Configuration and constants
    - auth: Cursor authentication manager
    - cache: Model metadata cache
    - model_resolver: Dynamic model resolution
    - http_client: HTTP/2 client with retry logic
    - converters_core: Unified message format and payload builder
    - converters_openai: OpenAI format adapter
    - converters_anthropic: Anthropic format adapter
    - streaming_core: Cursor stream parsing and event conversion
    - streaming_openai: OpenAI SSE streaming
    - streaming_anthropic: Anthropic SSE streaming
    - parsers: ConnectRPC stream parser
    - protobuf: Manual protobuf encoding/decoding
    - thinking_split: Thinking/reasoning stream splitter
    - redacted_tools: DeepSeek-native tool call extraction
    - bracket_tools: Bracket-format tool call extraction
    - tokenizer: Token counting (tiktoken)
    - utils: Header building, ID generation, platform info
    - checksum: Cursor checksum computation
    - exceptions: Exception handlers
"""

from cursor.config import APP_VERSION as __version__

__author__ = "Jwadow"

# Main components
from cursor.auth import CursorAuthManager
from cursor.cache import ModelInfoCache
from cursor.http_client import CursorHttpClient
from cursor.model_resolver import ModelResolver
from cursor.routes_openai import router as openai_router
from cursor.routes_anthropic import router as anthropic_router

# Configuration
from cursor.config import (
    PROXY_API_KEY,
    APP_VERSION,
    CURSOR_API_HOST,
    MODEL_ALIASES,
)

# Converters
from cursor.converters_core import (
    UnifiedMessage,
    UnifiedTool,
    BuildResult,
    extract_text_content,
    build_cursor_payload,
)

# Streaming
from cursor.streaming_core import (
    CursorEvent,
    parse_cursor_stream,
    FirstTokenTimeoutError,
)

# Exceptions
from cursor.exceptions import (
    validation_exception_handler,
    sanitize_validation_errors,
)

__all__ = [
    "__version__",
    "CursorAuthManager",
    "ModelInfoCache",
    "CursorHttpClient",
    "ModelResolver",
    "openai_router",
    "anthropic_router",
    "PROXY_API_KEY",
    "APP_VERSION",
    "CURSOR_API_HOST",
    "MODEL_ALIASES",
    "UnifiedMessage",
    "UnifiedTool",
    "BuildResult",
    "extract_text_content",
    "build_cursor_payload",
    "CursorEvent",
    "parse_cursor_stream",
    "FirstTokenTimeoutError",
    "validation_exception_handler",
    "sanitize_validation_errors",
]