"""
Core streaming logic for parsing Cursor ConnectRPC streams.

Provides unified event parsing and first-token timeout handling.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional

import httpx
from loguru import logger

from cursor.config import FIRST_TOKEN_TIMEOUT
from cursor.parsers import ConnectRpcStreamParser, StreamEvent


class FirstTokenTimeoutError(Exception):
    """Raised when the first token is not received within the timeout."""
    pass


@dataclass
class CursorEvent:
    """Unified event from Cursor stream."""
    type: str  # "content", "thinking", "tool_use", "usage", "end", "error"
    content: Optional[str] = None
    thinking_content: Optional[str] = None
    tool_use: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[Any] = None


async def parse_cursor_stream(
    response: httpx.Response,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
) -> AsyncGenerator[CursorEvent, None]:
    """
    Parses Cursor ConnectRPC stream and yields unified CursorEvent objects.

    Args:
        response: HTTP response with streaming data
        first_token_timeout: Timeout for first token (seconds)

    Yields:
        CursorEvent objects

    Raises:
        FirstTokenTimeoutError: If first token not received within timeout
    """
    parser = ConnectRpcStreamParser()
    first_token_received = False

    async def read_stream():
        nonlocal first_token_received
        async for chunk in response.aiter_bytes():
            events = parser.feed(chunk)
            for event in events:
                if not first_token_received and event.type in ("content", "thinking"):
                    first_token_received = True
                yield event

    stream_iter = read_stream()

    # Wait for first token with timeout
    try:
        first_event = await asyncio.wait_for(
            stream_iter.__anext__(),
            timeout=first_token_timeout
        )

        # Yield the first event
        yield _convert_event(first_event)

    except asyncio.TimeoutError:
        raise FirstTokenTimeoutError(
            f"First token not received within {first_token_timeout}s"
        )
    except StopAsyncIteration:
        return

    # Continue reading remaining events (no timeout per-event)
    async for event in stream_iter:
        yield _convert_event(event)


def _convert_event(event: StreamEvent) -> CursorEvent:
    """Convert a parser StreamEvent to a CursorEvent."""
    if event.type == "content":
        return CursorEvent(type="content", content=event.data)
    elif event.type == "thinking":
        return CursorEvent(type="thinking", thinking_content=event.data)
    elif event.type == "tool_use":
        return CursorEvent(type="tool_use", tool_use=event.data)
    elif event.type == "usage":
        return CursorEvent(type="usage", usage=event.data)
    elif event.type == "error":
        return CursorEvent(type="error", error=event.data)
    else:
        return CursorEvent(type=event.type)


async def stream_with_first_token_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    stream_processor: Callable[[httpx.Response], AsyncGenerator[str, None]],
    max_retries: int = 3,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    on_http_error: Optional[Callable] = None,
    on_all_retries_failed: Optional[Callable] = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming with automatic retry on first token timeout.

    If model doesn't respond within first_token_timeout seconds,
    request is cancelled and retried. Maximum max_retries attempts.

    Args:
        make_request: Function to create new HTTP request
        stream_processor: Function to process response into SSE chunks
        max_retries: Maximum number of attempts
        first_token_timeout: First token wait timeout (seconds)
        on_http_error: Callback for HTTP errors, returns exception to raise
        on_all_retries_failed: Callback when all retries exhausted, returns exception to raise

    Yields:
        Strings in SSE format
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = await make_request()

            # Check HTTP status
            if response.status_code != 200:
                try:
                    error_content = await response.aread()
                    error_text = error_content.decode("utf-8", errors="replace")
                except Exception:
                    error_text = f"HTTP {response.status_code}"

                if on_http_error:
                    raise on_http_error(response.status_code, error_text)
                raise Exception(f"HTTP {response.status_code}: {error_text}")

            # Process stream
            async for chunk in stream_processor(response):
                yield chunk

            return  # Success

        except FirstTokenTimeoutError as e:
            last_error = e
            logger.warning(
                f"First token timeout ({first_token_timeout}s), "
                f"attempt {attempt + 1}/{max_retries}"
            )
            # Close the response before retrying
            try:
                await response.aclose()
            except Exception:
                pass
            continue

        except Exception:
            raise

    # All retries exhausted
    if on_all_retries_failed:
        raise on_all_retries_failed(max_retries, first_token_timeout)
    raise last_error or Exception("All retries exhausted")
