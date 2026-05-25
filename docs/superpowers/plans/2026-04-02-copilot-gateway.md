# Copilot Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a GitHub Copilot gateway in `copilot/` that proxies OpenAI and Anthropic compatible API requests to the GitHub Copilot upstream API.

**Architecture:** Mirrors the existing `cursor/` gateway structure. Two-step auth (GitHub token → Copilot token). Upstream API is OpenAI-compatible, so the OpenAI path is near-passthrough. Anthropic path requires bidirectional format conversion.

**Tech Stack:** Python 3.10+, FastAPI, httpx, loguru, tiktoken, pydantic

**Spec:** `docs/superpowers/specs/2026-04-02-copilot-gateway-design.md`

**Reference implementation:** `cursor/` directory (follow its patterns exactly)

## File Structure

```
copilot/
├── __init__.py              — Package marker
├── config.py                — Environment variables and constants
├── auth.py                  — CopilotAuthManager (GitHub token → Copilot token)
├── exceptions.py            — Validation error handler
├── utils.py                 — ID generators, header builder
├── cache.py                 — Model metadata cache
├── tokenizer.py             — Token counting
├── model_resolver.py        — Model name normalization
├── models_openai.py         — Pydantic models for OpenAI API
├── models_anthropic.py      — Pydantic models for Anthropic API
├── converters_core.py       — Unified format → OpenAI request body
├── converters_openai.py     — OpenAI client request → unified format
├── converters_anthropic.py  — Anthropic client request → unified format
├── http_client.py           — HTTP client with retry logic
├── streaming_core.py        — Parse upstream OpenAI SSE → CopilotEvent
├── streaming_openai.py      — CopilotEvent → OpenAI SSE for client
├── streaming_anthropic.py   — CopilotEvent → Anthropic SSE for client
├── routes_openai.py         — /v1/models, /v1/chat/completions
└── routes_anthropic.py      — /v1/messages

main_copilot.py              — Entry point (project root)
.env.copilot.example         — Example env file (project root)
requirements_copilot.txt     — Dependencies (project root)
```

## Tasks

### Task 1: Scaffolding — config, env, requirements

**Files:** Create `copilot/__init__.py`, `copilot/config.py`, `requirements_copilot.txt`, `.env.copilot.example`

- [ ] Create empty `copilot/__init__.py`
- [ ] Create `requirements_copilot.txt` — same as `requirements.txt` (fastapi, uvicorn[standard], httpx, loguru, python-dotenv, tiktoken, pytest, pytest-asyncio, hypothesis)
- [ ] Create `.env.copilot.example` — modeled on `.env.cursor.example` with: `PROXY_API_KEY`, `GITHUB_TOKEN`, `COPILOT_VSCODE_DB_FILE`, `COPILOT_SERVER_HOST/PORT` (default 8002), `VPN_PROXY_URL`, `FIRST_TOKEN_TIMEOUT`, `STREAMING_READ_TIMEOUT`, `LOG_LEVEL`
- [ ] Create `copilot/config.py` — copy `cursor/config.py` structure, change:
  - Load `.env.copilot` instead of `.env.cursor`
  - Default port 8002
  - Replace Cursor API settings with: `GITHUB_API_HOST = "https://api.github.com"`, `COPILOT_TOKEN_ENDPOINT`, `COPILOT_API_HOST = "https://api.individual.githubcopilot.com"`, `COPILOT_CHAT_ENDPOINT = "/chat/completions"`, `COPILOT_MODELS_ENDPOINT = "/models"`
  - Replace Cursor credentials with: `GITHUB_TOKEN`, `COPILOT_VSCODE_DB_PATH` (auto-detect VS Code paths, not Cursor paths)
  - `TOKEN_REFRESH_THRESHOLD = 300` (5 min, copilot token is ~30 min)
  - `FALLBACK_MODELS`: gpt-4o, gpt-4o-mini, claude-sonnet-4, claude-3.5-sonnet, o1, o3-mini
  - `APP_TITLE = "Copilot Gateway"`, `APP_DESCRIPTION` updated
  - Remove: `CURSOR_*` settings, `CURSOR_GHOST_MODE`, checksum, protobuf, debug settings
- [ ] Commit: `git commit -m "feat(copilot): add project scaffolding and config"`

### Task 2: Shared modules — exceptions, utils, cache, tokenizer, models, model_resolver

**Files:** Create `copilot/exceptions.py`, `copilot/utils.py`, `copilot/cache.py`, `copilot/tokenizer.py`, `copilot/models_openai.py`, `copilot/models_anthropic.py`, `copilot/model_resolver.py`

- [ ] Create `copilot/exceptions.py` — copy `cursor/exceptions.py` exactly, only change import path from `cursor.config` to nothing (no debug logger needed)
- [ ] Create `copilot/cache.py` — copy `cursor/cache.py` exactly, change import from `cursor.config` to `copilot.config`
- [ ] Create `copilot/tokenizer.py` — copy `cursor/tokenizer.py` exactly (no import changes needed, it's self-contained)
- [ ] Create `copilot/models_openai.py` — copy `cursor/models_openai.py` exactly (no import changes needed)
- [ ] Create `copilot/models_anthropic.py` — copy `cursor/models_anthropic.py` exactly (no import changes needed)
- [ ] Create `copilot/model_resolver.py` — copy `cursor/model_resolver.py`, change imports from `cursor.cache`/`cursor.config` to `copilot.cache`/`copilot.config`
- [ ] Create `copilot/utils.py` — new file, simpler than cursor's (no checksum, no ConnectRPC headers):

```python
"""
Utility functions for Copilot Gateway.
"""

import uuid


def get_copilot_headers(copilot_token: str) -> dict:
    """Builds headers for GitHub Copilot API requests."""
    return {
        "Authorization": f"Bearer {copilot_token}",
        "Content-Type": "application/json",
        "Editor-Version": "vscode/1.96.0",
        "Copilot-Integration-Id": "vscode-chat",
        "Openai-Intent": "conversation-panel",
        "X-Request-Id": str(uuid.uuid4()),
    }


def generate_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_conversation_id() -> str:
    return str(uuid.uuid4())


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def generate_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"
```

- [ ] Commit: `git commit -m "feat(copilot): add shared modules"`

### Task 3: Authentication — auth.py

**Files:** Create `copilot/auth.py`

- [ ] Create `copilot/auth.py` with `CopilotAuthManager`:

```python
"""
Authentication manager for GitHub Copilot API.

Two-step flow:
1. Get GitHub token (env var or VS Code SQLite)
2. Exchange for short-lived Copilot token via GitHub API
"""

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from copilot.config import (
    GITHUB_TOKEN,
    COPILOT_VSCODE_DB_PATH,
    COPILOT_TOKEN_ENDPOINT,
    COPILOT_API_HOST,
    TOKEN_REFRESH_THRESHOLD,
)


class CopilotAuthManager:
    """
    Manages GitHub Copilot authentication tokens.

    Token sources (priority order):
    1. Environment variable (GITHUB_TOKEN)
    2. VS Code SQLite database (state.vscdb)
    """

    def __init__(
        self,
        github_token: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self._github_token = github_token or GITHUB_TOKEN or None
        self._db_path = db_path or COPILOT_VSCODE_DB_PATH or None

        # Load from SQLite if token not provided via env
        if not self._github_token and self._db_path:
            self._load_from_sqlite(self._db_path)

        # Copilot token state
        self._copilot_token: Optional[str] = None
        self._copilot_token_expires_at: float = 0
        self._copilot_api_host: str = COPILOT_API_HOST
        self._lock = asyncio.Lock()

        if self._github_token:
            preview = self._github_token[:12] + "..."
            logger.info(f"Copilot auth initialized: github_token={preview}")
        else:
            logger.warning("Copilot auth: no GitHub token available")

    def _load_from_sqlite(self, db_path: str) -> None:
        """Load GitHub token from VS Code's state.vscdb."""
        try:
            path = Path(db_path).expanduser()
            if not path.exists():
                logger.warning(f"VS Code database not found: {db_path}")
                return

            uri = f"file:{path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            cursor = conn.cursor()

            # Try known keys for GitHub/Copilot token
            for key in [
                "github.copilot.chat.token",
                "github.copilot.token",
                "github-enterprise.copilot.token",
            ]:
                cursor.execute(
                    "SELECT value FROM ItemTable WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                if row and row[0]:
                    raw = row[0]
                    # Token may be JSON string with quotes
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = raw[1:-1]
                    self._github_token = raw
                    logger.info(f"Loaded GitHub token from SQLite (key={key})")
                    break

            conn.close()
        except sqlite3.OperationalError as e:
            logger.error(f"SQLite error reading VS Code DB: {e}")
        except Exception as e:
            logger.error(f"Error loading GitHub token: {e}")

    async def get_copilot_token(self) -> str:
        """
        Returns a valid Copilot API token.
        Refreshes automatically if expired or about to expire.

        Raises:
            ValueError: If no GitHub token is available
        """
        async with self._lock:
            now = time.time()
            if (
                self._copilot_token
                and self._copilot_token_expires_at - now > TOKEN_REFRESH_THRESHOLD
            ):
                return self._copilot_token

            # Need to refresh
            await self._exchange_token()

            if not self._copilot_token:
                raise ValueError(
                    "Failed to obtain Copilot token. "
                    "Check your GitHub token or VS Code login."
                )
            return self._copilot_token

    async def _exchange_token(self) -> None:
        """Exchange GitHub token for Copilot token."""
        if not self._github_token:
            # Try reloading from SQLite
            if self._db_path:
                self._load_from_sqlite(self._db_path)
            if not self._github_token:
                raise ValueError(
                    "No GitHub token available. "
                    "Set GITHUB_TOKEN or log in to VS Code with Copilot."
                )

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    COPILOT_TOKEN_ENDPOINT,
                    headers={
                        "Authorization": f"token {self._github_token}",
                        "Accept": "application/json",
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    self._copilot_token = data.get("token")
                    self._copilot_token_expires_at = data.get("expires_at", 0)

                    # Update API host if provided
                    endpoints = data.get("endpoints", {})
                    if endpoints.get("api"):
                        self._copilot_api_host = endpoints["api"]

                    logger.info(
                        f"Copilot token obtained, expires_at={self._copilot_token_expires_at}, "
                        f"api_host={self._copilot_api_host}"
                    )
                elif response.status_code == 401:
                    logger.error("GitHub token is invalid or expired (401)")
                    # Try reloading from SQLite
                    if self._db_path:
                        self._load_from_sqlite(self._db_path)
                else:
                    logger.error(
                        f"Failed to get Copilot token: HTTP {response.status_code} "
                        f"{response.text[:200]}"
                    )
        except Exception as e:
            logger.error(f"Error exchanging token: {e}")

    async def force_refresh(self) -> None:
        """Force token refresh (e.g., after 401 from upstream)."""
        async with self._lock:
            self._copilot_token = None
            self._copilot_token_expires_at = 0
            await self._exchange_token()

    @property
    def api_host(self) -> str:
        """Copilot API base URL (from token response)."""
        return self._copilot_api_host
```

- [ ] Commit: `git commit -m "feat(copilot): add authentication manager"`

### Task 4: HTTP client

**Files:** Create `copilot/http_client.py`

- [ ] Create `copilot/http_client.py` — adapted from `cursor/http_client.py`:
  - HTTP/1.1 (not HTTP/2, no ConnectRPC)
  - `request_with_retry` sends JSON body (dict), not bytes
  - On 401: call `auth_manager.force_refresh()` and retry
  - Headers from `get_copilot_headers(token)` where token comes from `auth_manager.get_copilot_token()`
  - Same retry logic: 429 exponential backoff, 5xx exponential backoff, timeout exponential backoff

```python
"""
HTTP client for Copilot API with retry logic.
"""

import asyncio
import json
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from copilot.config import (
    MAX_RETRIES,
    BASE_RETRY_DELAY,
    FIRST_TOKEN_MAX_RETRIES,
    STREAMING_READ_TIMEOUT,
)
from copilot.auth import CopilotAuthManager
from copilot.utils import get_copilot_headers


class CopilotHttpClient:
    def __init__(
        self,
        auth_manager: CopilotAuthManager,
        shared_client: Optional[httpx.AsyncClient] = None,
    ):
        self.auth_manager = auth_manager
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client

    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        if self._shared_client is not None:
            return self._shared_client

        if self.client is None or self.client.is_closed:
            if stream:
                timeout_config = httpx.Timeout(
                    connect=30.0, read=STREAMING_READ_TIMEOUT,
                    write=30.0, pool=30.0,
                )
            else:
                timeout_config = httpx.Timeout(timeout=300.0)

            self.client = httpx.AsyncClient(
                timeout=timeout_config, follow_redirects=True,
            )
        return self.client

    async def close(self) -> None:
        if not self._owns_client:
            return
        if self.client and not self.client.is_closed:
            try:
                await self.client.aclose()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")

    async def request_with_retry(
        self,
        method: str,
        url: str,
        data: Dict[str, Any],
        stream: bool = False,
    ) -> httpx.Response:
        max_retries = FIRST_TOKEN_MAX_RETRIES if stream else MAX_RETRIES
        client = await self._get_client(stream=stream)
        last_error = None

        for attempt in range(max_retries):
            try:
                token = await self.auth_manager.get_copilot_token()
                headers = get_copilot_headers(token)

                if stream:
                    req = client.build_request(
                        method, url,
                        json=data,
                        headers=headers,
                    )
                    response = await client.send(req, stream=True)
                else:
                    response = await client.request(
                        method, url,
                        json=data,
                        headers=headers,
                    )

                if response.status_code == 200:
                    return response

                if response.status_code == 401:
                    logger.warning(f"401 from Copilot API, refreshing token (attempt {attempt + 1}/{max_retries})")
                    await self.auth_manager.force_refresh()
                    continue

                if response.status_code == 429:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"429 rate limit, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue

                if 500 <= response.status_code < 600:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"{response.status_code} server error, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue

                return response

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Timeout: {e} - waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Timeout: {e} - no more retries")

            except httpx.RequestError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Request error: {e} - waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request error: {e} - no more retries")

        error_msg = str(last_error) if last_error else "Unknown error"
        status = 504 if stream else 502
        raise HTTPException(
            status_code=status,
            detail=f"Request failed after {max_retries} attempts: {error_msg}",
        )

    async def __aenter__(self) -> "CopilotHttpClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
```

- [ ] Commit: `git commit -m "feat(copilot): add HTTP client with retry logic"`

### Task 5: Converters — core, openai, anthropic

**Files:** Create `copilot/converters_core.py`, `copilot/converters_openai.py`, `copilot/converters_anthropic.py`

- [ ] Create `copilot/converters_core.py` — different from cursor's version. Cursor builds protobuf; Copilot builds an OpenAI JSON dict. Keep `UnifiedMessage`, `UnifiedTool`, `extract_text_content`, `extract_images_from_content`, `_merge_consecutive_messages` from cursor. Replace `build_cursor_payload` with:

```python
@dataclass
class BuildResult:
    """Result of building a Copilot payload."""
    payload: Dict[str, Any]  # OpenAI-format JSON dict (not bytes)
    model_id: str
    message_count: int


def build_copilot_payload(
    messages: List[UnifiedMessage],
    system_prompt: str,
    model_id: str,
    tools: Optional[List[UnifiedTool]] = None,
    conversation_id: str = "",
    **kwargs,
) -> BuildResult:
    """Builds OpenAI-format JSON payload for Copilot API."""
    if not messages:
        raise ValueError("No messages to send")

    # Ensure last message is from user
    if messages[-1].role != "user":
        messages.append(UnifiedMessage(role="user", content="(empty)"))

    # Build OpenAI messages list
    openai_messages = []

    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        openai_msg: Dict[str, Any] = {"role": msg.role, "content": msg.content or ""}

        if msg.tool_calls:
            openai_msg["tool_calls"] = msg.tool_calls
            if not openai_msg["content"]:
                openai_msg["content"] = None

        if msg.tool_results:
            # Convert tool results to separate tool messages
            for tr in msg.tool_results:
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_use_id", ""),
                    "content": tr.get("content", ""),
                })
            if openai_msg["content"]:
                openai_messages.append(openai_msg)
            continue

        openai_messages.append(openai_msg)

    # Merge consecutive same-role messages
    openai_messages = _merge_consecutive_messages(openai_messages)

    payload = {
        "model": model_id,
        "messages": openai_messages,
        "stream": True,
    }

    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema or {},
                },
            }
            for t in tools
        ]

    return BuildResult(
        payload=payload,
        model_id=model_id,
        message_count=len(openai_messages),
    )
```

- [ ] Create `copilot/converters_openai.py` — copy `cursor/converters_openai.py`, change all imports from `cursor.*` to `copilot.*`. The `build_cursor_payload` function returns `result.payload` (now a dict, not bytes).
- [ ] Create `copilot/converters_anthropic.py` — copy `cursor/converters_anthropic.py`, change all imports from `cursor.*` to `copilot.*`. The `anthropic_to_cursor` function renamed to `anthropic_to_copilot`, returns `result.payload` (dict).
- [ ] Commit: `git commit -m "feat(copilot): add format converters"`

### Task 6: Streaming — core, openai, anthropic

**Files:** Create `copilot/streaming_core.py`, `copilot/streaming_openai.py`, `copilot/streaming_anthropic.py`

- [ ] Create `copilot/streaming_core.py` — parses standard OpenAI SSE (not ConnectRPC). Much simpler than cursor's:

```python
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
                # May contain usage only
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
```

- [ ] Create `copilot/streaming_openai.py` — near-passthrough since upstream is already OpenAI SSE. Adapted from `cursor/streaming_openai.py`:
  - Use `parse_copilot_stream` instead of `parse_cursor_stream`
  - `CopilotEvent` has `content`, `reasoning_content`, `tool_calls`, `finish_reason`
  - For content events: yield OpenAI chunk with `delta.content`
  - For reasoning_content events: yield OpenAI chunk with `delta.reasoning_content`
  - For tool_call events: accumulate tool calls, yield at end
  - For done events: yield final chunk with finish_reason and usage
  - Keep `collect_stream_response` for non-streaming mode
  - Keep `stream_with_first_token_retry` wrapper

- [ ] Create `copilot/streaming_anthropic.py` — copy `cursor/streaming_anthropic.py`, change imports from `cursor.*` to `copilot.*`, use `parse_copilot_stream` and `CopilotEvent`. Adapt event handling:
  - `CopilotEvent.type == "content"` with `event.content` → text block delta
  - `CopilotEvent.type == "content"` with `event.reasoning_content` → thinking block delta
  - `CopilotEvent.type == "tool_call"` → tool_use blocks
  - `CopilotEvent.type == "done"` → close blocks, message_delta, message_stop

- [ ] Commit: `git commit -m "feat(copilot): add streaming parsers and formatters"`

### Task 7: Routes — openai, anthropic

**Files:** Create `copilot/routes_openai.py`, `copilot/routes_anthropic.py`

- [ ] Create `copilot/routes_openai.py` — copy `cursor/routes_openai.py`, change:
  - All imports from `cursor.*` to `copilot.*`
  - `CursorAuthManager` → `CopilotAuthManager`
  - `CursorHttpClient` → `CopilotHttpClient`
  - `build_cursor_payload` → `build_copilot_payload` (from `copilot.converters_openai`)
  - URL: `f"{auth_manager.api_host}{COPILOT_CHAT_ENDPOINT}"` (not hardcoded)
  - `http_client.request_with_retry("POST", url, copilot_payload, stream=True)` — payload is dict now, not bytes
  - `stream_cursor_to_openai` → `stream_copilot_to_openai` (from `copilot.streaming_openai`)
  - `collect_stream_response` from `copilot.streaming_openai`
  - Models endpoint: `owned_by="copilot"`

- [ ] Create `copilot/routes_anthropic.py` — copy `cursor/routes_anthropic.py`, change:
  - All imports from `cursor.*` to `copilot.*`
  - `CursorAuthManager` → `CopilotAuthManager`
  - `CursorHttpClient` → `CopilotHttpClient`
  - `anthropic_to_cursor` → `anthropic_to_copilot` (from `copilot.converters_anthropic`)
  - URL: `f"{auth_manager.api_host}{COPILOT_CHAT_ENDPOINT}"`
  - `stream_cursor_to_anthropic` → `stream_copilot_to_anthropic`
  - `collect_anthropic_response` from `copilot.streaming_anthropic`

- [ ] Commit: `git commit -m "feat(copilot): add OpenAI and Anthropic routes"`

### Task 8: Entry point — main_copilot.py

**Files:** Create `main_copilot.py` (project root)

- [ ] Create `main_copilot.py` — copy `main_cursor.py`, change:
  - All imports from `cursor.*` to `copilot.*`
  - `CursorAuthManager` → `CopilotAuthManager`
  - Load `.env.copilot` config
  - Default port 8002
  - Validate: check `GITHUB_TOKEN` or `COPILOT_VSCODE_DB_PATH` exists
  - HTTP/1.1 client (remove `http2=True`)
  - Model loading: `GET {auth_manager.api_host}/models` with copilot headers, parse JSON response for model list, fallback to `FALLBACK_MODELS`
  - Startup banner: 🐙 emoji, "Copilot Gateway"
  - `UVICORN_LOG_CONFIG` handler class: `main_copilot.InterceptHandler`
  - `uvicorn.run("main_copilot:app", ...)`

```python
"""
Copilot Gateway - OpenAI/Anthropic-compatible interface for GitHub Copilot API.

Usage:
    python main_copilot.py
    python main_copilot.py --port 8002
"""

import argparse
import logging
import sys
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from copilot.config import (
    APP_TITLE, APP_DESCRIPTION, APP_VERSION,
    GITHUB_TOKEN, COPILOT_VSCODE_DB_PATH,
    COPILOT_MODELS_ENDPOINT,
    PROXY_API_KEY, LOG_LEVEL,
    SERVER_HOST, SERVER_PORT,
    DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT,
    STREAMING_READ_TIMEOUT,
    MODEL_ALIASES, HIDDEN_FROM_LIST, FALLBACK_MODELS,
    VPN_PROXY_URL,
)
from copilot.auth import CopilotAuthManager
from copilot.cache import ModelInfoCache
from copilot.model_resolver import ModelResolver
from copilot.routes_openai import router as openai_router
from copilot.routes_anthropic import router as anthropic_router
from copilot.exceptions import validation_exception_handler
from copilot.utils import get_copilot_headers


# --- Loguru Configuration ---
logger.remove()
logger.add(
    sys.stderr, level=LOG_LEVEL, colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info:
            exc_type = record.exc_info[0]
            if exc_type is not None and exc_type.__name__ in ("CancelledError", "KeyboardInterrupt"):
                return
        msg = record.getMessage()
        if any(exc in msg for exc in ("CancelledError", "KeyboardInterrupt")):
            return
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging_intercept():
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

setup_logging_intercept()


# --- VPN/Proxy ---
if VPN_PROXY_URL:
    proxy_url = VPN_PROXY_URL if "://" in VPN_PROXY_URL else f"http://{VPN_PROXY_URL}"
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url
    no_proxy = os.environ.get("NO_PROXY", "")
    local = "127.0.0.1,localhost"
    os.environ["NO_PROXY"] = f"{no_proxy},{local}" if no_proxy else local
    logger.info(f"Proxy configured: {proxy_url}")


def validate_configuration() -> None:
    has_token = bool(GITHUB_TOKEN)
    has_db = bool(COPILOT_VSCODE_DB_PATH)
    if has_db:
        from pathlib import Path
        if not Path(COPILOT_VSCODE_DB_PATH).expanduser().exists():
            has_db = False
            logger.warning(f"VS Code database not found: {COPILOT_VSCODE_DB_PATH}")
    if not has_token and not has_db:
        logger.error("")
        logger.error("=" * 60)
        logger.error("  CONFIGURATION ERROR")
        logger.error("=" * 60)
        logger.error("  No GitHub Copilot credentials configured!")
        logger.error("")
        logger.error("  Options:")
        logger.error("    1. Set GITHUB_TOKEN in .env.copilot")
        logger.error("    2. Log in to VS Code with GitHub Copilot")
        logger.error("    3. Set COPILOT_VSCODE_DB_FILE to your state.vscdb path")
        logger.error("")
        logger.error("  See .env.copilot.example for details.")
        logger.error("=" * 60)
        logger.error("")
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Copilot Gateway...")

    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0)
    timeout = httpx.Timeout(connect=30.0, read=STREAMING_READ_TIMEOUT, write=30.0, pool=30.0)
    app.state.http_client = httpx.AsyncClient(limits=limits, timeout=timeout, follow_redirects=True)
    logger.info("Shared HTTP client created")

    app.state.auth_manager = CopilotAuthManager(
        db_path=COPILOT_VSCODE_DB_PATH if COPILOT_VSCODE_DB_PATH else None,
    )

    app.state.model_cache = ModelInfoCache()

    # Try to fetch models from Copilot API
    logger.info("Loading models from Copilot API...")
    try:
        token = await app.state.auth_manager.get_copilot_token()
        headers = get_copilot_headers(token)
        url = f"{app.state.auth_manager.api_host}{COPILOT_MODELS_ENDPOINT}"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                models_list = data.get("data", data) if isinstance(data, dict) else data
                if isinstance(models_list, list):
                    models_data = [
                        {"modelId": m.get("id", m.get("modelId", ""))}
                        for m in models_list if isinstance(m, dict)
                    ]
                    if models_data:
                        await app.state.model_cache.update(models_data)
                        logger.info(f"Loaded {len(models_data)} models from Copilot API")
                    else:
                        raise Exception("Empty model list")
                else:
                    raise Exception("Unexpected response format")
            else:
                raise Exception(f"HTTP {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not fetch models from Copilot API: {e}")
        logger.info("Using fallback model list")
        await app.state.model_cache.update(FALLBACK_MODELS)

    app.state.model_resolver = ModelResolver(
        cache=app.state.model_cache,
        aliases=MODEL_ALIASES,
        hidden_from_list=HIDDEN_FROM_LIST,
    )
    logger.info(f"Model resolver initialized with {app.state.model_cache.size} models")

    yield

    logger.info("Shutting down...")
    try:
        await app.state.http_client.aclose()
    except Exception as e:
        logger.warning(f"Error closing HTTP client: {e}")


app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.include_router(openai_router)
app.include_router(anthropic_router)

UVICORN_LOG_CONFIG = {
    "version": 1, "disable_existing_loggers": False,
    "handlers": {"default": {"class": "main_copilot.InterceptHandler"}},
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_TITLE} - {APP_DESCRIPTION}")
    parser.add_argument("-H", "--host", type=str, default=None, help=f"Server host (default: {DEFAULT_SERVER_HOST})")
    parser.add_argument("-p", "--port", type=int, default=None, help=f"Server port (default: {DEFAULT_SERVER_PORT})")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {APP_VERSION}")
    return parser.parse_args()


def print_startup_banner(host: str, port: int) -> None:
    GREEN, WHITE, BOLD, DIM, RESET = "\033[92m", "\033[97m", "\033[1m", "\033[2m", "\033[0m"
    display_host = "localhost" if host == "0.0.0.0" else host
    url = f"http://{display_host}:{port}"
    print()
    print(f"  {WHITE}{BOLD}🐙  {APP_TITLE} v{APP_VERSION}{RESET}")
    print()
    print(f"  {WHITE}Server running at:{RESET}")
    print(f"  {GREEN}{BOLD}➜  {url}{RESET}")
    print()
    print(f"  {DIM}API Docs:      {url}/docs{RESET}")
    print(f"  {DIM}Health Check:  {url}/health{RESET}")
    print()
    print(f"  {DIM}{'─' * 48}{RESET}")
    print(f"  {WHITE}OpenAI API:     {url}/v1/chat/completions{RESET}")
    print(f"  {WHITE}Anthropic API:  {url}/v1/messages{RESET}")
    print(f"  {DIM}{'─' * 48}{RESET}")
    print()


if __name__ == "__main__":
    import uvicorn
    validate_configuration()
    args = parse_cli_args()
    final_host = args.host if args.host else SERVER_HOST
    final_port = args.port if args.port else SERVER_PORT
    print_startup_banner(final_host, final_port)
    logger.info(f"Starting Uvicorn on {final_host}:{final_port}...")
    uvicorn.run("main_copilot:app", host=final_host, port=final_port, log_config=UVICORN_LOG_CONFIG)
```

- [ ] Commit: `git commit -m "feat(copilot): add entry point main_copilot.py"`

### Task 9: Smoke test

- [ ] Run `python main_copilot.py` and verify it starts without import errors
- [ ] Run `curl http://localhost:8002/health` and verify response
- [ ] Run `curl http://localhost:8002/v1/models -H "Authorization: Bearer my-super-secret-password-123"` and verify model list
- [ ] If credentials are available, test a simple chat completion:

```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Authorization: Bearer my-super-secret-password-123" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

- [ ] Test Anthropic endpoint:

```bash
curl http://localhost:8002/v1/messages \
  -H "x-api-key: my-super-secret-password-123" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello!"}]}'
```

- [ ] Fix any issues found during smoke testing
- [ ] Final commit: `git commit -m "feat(copilot): copilot gateway complete"`

