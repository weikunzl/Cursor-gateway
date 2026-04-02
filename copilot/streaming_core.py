"""
Core streaming logic for parsing Copilot (OpenAI SSE) streams.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional

import httpx
from loguru import logger

from copilot.config import FIRST_TOKEN_TIMEOUT


class FirstTokenTimeoutError(Exception):
    pass


@dataclass
class CopilotEvent:
    """Unified event from Copilot stream."""
    type: str  # "content", "tool_call", "usage", "done", "error"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    error: Optional[Any] = None


async def parse_copilot_stream(
    response: httpx.Response,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
) -> AsyncGenerator[CopilotEvent, None]:
    """Parses OpenAI SSE stream from Copilot API into CopilotEvent objects."""
    first_token_received = False

    async def read_lines():
        nonlocal first_token_received
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                return
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                if "usage" in chunk:
                    yield CopilotEvent(type="usage", usage=chunk["usage"])
                continue

            delta = choices[0].get("delta", {})
            finish = choices[0].get("finish_reason")

            if "content" in delta and delta["content"]:
                if not first_token_received:
                    first_token_received = True
                yield CopilotEvent(type="content", content=delta["content"])

            if "reasoning_content" in delta and delta["reasoning_content"]:
                if not first_token_received:
                    first_token_received = True
                yield CopilotEvent(
                    type="content",
                    content=None,
                    reasoning_content=delta["reasoning_content"],
                )

            if "tool_calls" in delta:
                yield CopilotEvent(type="tool_call", tool_calls=delta["tool_calls"])

            if finish:
                usage = chunk.get("usage")
                yield CopilotEvent(type="done", finish_reason=finish, usage=usage)

    stream_iter = read_lines()

    try:
        first_event = await asyncio.wait_for(
            stream_iter.__anext__(), timeout=first_token_timeout,
        )
        yield first_event
    except asyncio.TimeoutError:
        raise FirstTokenTimeoutError(
            f"First token not received within {first_token_timeout}s"
        )
    except StopAsyncIteration:
        return

    async for event in stream_iter:
        yield event


async def stream_with_first_token_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    stream_processor: Callable[[httpx.Response], AsyncGenerator[str, None]],
    max_retries: int = 3,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    on_http_error: Optional[Callable] = None,
    on_all_retries_failed: Optional[Callable] = None,
) -> AsyncGenerator[str, None]:
    """Streaming with automatic retry on first token timeout."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = await make_request()
            if response.status_code != 200:
                try:
                    error_content = await response.aread()
                    error_text = error_content.decode("utf-8", errors="replace")
                except Exception:
                    error_text = f"HTTP {response.status_code}"
                if on_http_error:
                    raise on_http_error(response.status_code, error_text)
                raise Exception(f"HTTP {response.status_code}: {error_text}")

            async for chunk in stream_processor(response):
                yield chunk
            return

        except FirstTokenTimeoutError as e:
            last_error = e
            logger.warning(
                f"First token timeout ({first_token_timeout}s), "
                f"attempt {attempt + 1}/{max_retries}"
            )
            try:
                await response.aclose()
            except Exception:
                pass
            continue
        except Exception:
            raise

    if on_all_retries_failed:
        raise on_all_retries_failed(max_retries, first_token_timeout)
    raise last_error or Exception("All retries exhausted")
