# Copilot Gateway Design Spec

## Overview

Create a GitHub Copilot gateway (`copilot/` directory) that proxies OpenAI and Anthropic compatible API requests to the GitHub Copilot upstream API. Follows the same module structure as the existing `kiro/` and `cursor/` gateways for consistency.

Entry point: `main_copilot.py`, default port 8002.

## Architecture

```
Client (OpenAI/Anthropic format)
    ↓
Route Handler (routes_openai.py / routes_anthropic.py)
    ↓
Converter (converters_openai.py / converters_anthropic.py)
    ↓
Unified Format (converters_core.py)
    ↓
OpenAI Request Body (for Copilot API)
    ↓
HTTP Client (http_client.py) → Copilot API (api.individual.githubcopilot.com)
    ↓
OpenAI SSE Response
    ↓
Stream Parser (streaming_core.py)
    ↓
Format Converter (streaming_openai.py / streaming_anthropic.py)
    ↓
Client Response (OpenAI/Anthropic format)
```

Key simplification vs kiro/cursor: the upstream Copilot API is already OpenAI-compatible, so the OpenAI path is near-passthrough and no binary protocol parsing (protobuf/AWS SSE) is needed.

## Module Structure

```
copilot/
├── __init__.py
├── config.py
├── auth.py
├── routes_openai.py
├── routes_anthropic.py
├── converters_core.py
├── converters_openai.py
├── converters_anthropic.py
├── streaming_core.py
├── streaming_openai.py
├── streaming_anthropic.py
├── http_client.py
├── models_openai.py
├── models_anthropic.py
├── model_resolver.py
├── cache.py
├── tokenizer.py
├── utils.py
└── exceptions.py
```

Plus entry point: `main_copilot.py` (project root).

## Authentication (auth.py)

Two-step authentication flow.

### Step 1: Obtain GitHub Token

Two sources, priority: environment variable > SQLite.

1. Environment variable `GITHUB_TOKEN` — user provides directly.
2. VS Code SQLite database (`state.vscdb`) — auto-detect from default paths:
   - macOS: `~/Library/Application Support/Code/User/globalStorage/state.vscdb`
   - Linux: `~/.config/Code/User/globalStorage/state.vscdb`
   - Allow override via `COPILOT_VSCODE_DB_FILE` env var.
   - Read from `ItemTable`, key containing the GitHub/Copilot token.
   - Open in read-only mode (`?mode=ro`).

### Step 2: Exchange for Copilot Token

- Endpoint: `GET https://api.github.com/copilot_internal/v2/token`
- Header: `Authorization: token <github_token>`
- Response JSON contains:
  - `token` — short-lived JWT (~30 min)
  - `expires_at` — Unix timestamp
  - `endpoints.api` — API base URL (e.g., `https://api.individual.githubcopilot.com`)

### Token Management

- `CopilotAuthManager` class (mirrors `KiroAuthManager` / `CursorAuthManager`).
- Cache copilot token in memory.
- Auto-refresh 5 minutes before expiration.
- Force refresh on 401 from upstream.
- `asyncio.Lock` for thread safety.
- On SQLite source: re-read from database if GitHub token refresh fails.

## Configuration (config.py)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PROXY_API_KEY` | Yes | — | Password to protect the proxy |
| `GITHUB_TOKEN` | No | — | GitHub OAuth token (overrides SQLite) |
| `COPILOT_VSCODE_DB_FILE` | No | auto-detect | Path to VS Code state.vscdb |
| `COPILOT_SERVER_HOST` | No | `0.0.0.0` | Server bind host |
| `COPILOT_SERVER_PORT` | No | `8002` | Server bind port |
| `VPN_PROXY_URL` | No | — | HTTP/SOCKS5 proxy URL |
| `FIRST_TOKEN_TIMEOUT` | No | `15` | Seconds to wait for first token |
| `FIRST_TOKEN_MAX_RETRIES` | No | `3` | Max retry attempts on first-token timeout |
| `STREAMING_READ_TIMEOUT` | No | `300` | Seconds between stream chunks |
| `LOG_LEVEL` | No | `INFO` | Log level |

## Routes

### OpenAI Endpoint (routes_openai.py)

- `GET /v1/models` — list available models
- `POST /v1/chat/completions` — chat completions

Flow:
1. Validate `PROXY_API_KEY` (Bearer token).
2. Resolve model name via `model_resolver`.
3. Pass request through `converters_openai.py` (light normalization — remove unsupported params).
4. Send to Copilot API via `http_client`.
5. Stream response back (near-passthrough for OpenAI SSE).

### Anthropic Endpoint (routes_anthropic.py)

- `POST /v1/messages` — messages API

Flow:
1. Validate API key (`x-api-key` or `Authorization`).
2. Resolve model name.
3. Convert Anthropic format → unified format (`converters_anthropic.py`).
4. Convert unified format → OpenAI request body (`converters_core.py`).
5. Send to Copilot API.
6. Convert OpenAI SSE → Anthropic SSE (`streaming_anthropic.py`).

## Format Conversion

### converters_core.py

Unified internal format (same concept as kiro/cursor):
- `UnifiedMessage` — role, content, tool_calls, tool_results, images.
- `UnifiedTool` — name, description, input_schema.
- `build_copilot_payload(messages, tools, model, system_prompt, **kwargs)` → OpenAI request dict.

### converters_openai.py

OpenAI client request → unified format. Thin layer:
- Extract system messages.
- Normalize message content formats.
- Pass through tools as-is (already OpenAI format).

### converters_anthropic.py

Anthropic client request → unified format:
- Extract `system` field (string or content block list).
- Convert content blocks: text, image, tool_use, tool_result.
- Map Anthropic tool definitions to unified format.

## Streaming

### streaming_core.py

Parse upstream Copilot SSE (standard OpenAI format: `data: {...}\n\n`):
- Parse into `CopilotEvent` objects with type: content, tool_call, usage, done.
- Handle `data: [DONE]` termination.
- First-token timeout support (default 15s) with retry.

### streaming_openai.py

CopilotEvent → OpenAI SSE for client. Near-passthrough:
- Yield upstream chunks directly.
- Supplement usage info if missing (tiktoken estimation).
- Generate `completion_id`.

### streaming_anthropic.py

CopilotEvent → Anthropic SSE for client:
- Emit event sequence: `message_start` → `content_block_start` → `content_block_delta`(s) → `content_block_stop` → `message_delta` → `message_stop`.
- Convert tool_use events to Anthropic tool_use blocks.
- Generate Anthropic-style IDs (`msg_xxx`).
- Calculate and report token usage.

## HTTP Client (http_client.py)

- Standard HTTP/1.1 via httpx.
- Required upstream headers:
  - `Authorization: Bearer <copilot_token>`
  - `Editor-Version: vscode/1.96.0`
  - `Copilot-Integration-Id: vscode-chat`
  - `Openai-Intent: conversation-panel`
  - `Content-Type: application/json`
- Retry logic:
  - 401: force token refresh, retry.
  - 429: exponential backoff (1s, 2s, 4s).
  - 5xx: exponential backoff.
  - Timeout: exponential backoff.
- Streaming: per-request client. Non-streaming: shared client with connection pooling.

## Model Resolution (model_resolver.py)

- Fetch available models from Copilot `/models` endpoint at startup.
- Fallback to pre-configured list: gpt-4o, claude-sonnet-4, o1, o3-mini, etc.
- Model name normalization consistent with kiro/cursor style.
- Cache with TTL (1 hour).

## Supporting Modules

- `models_openai.py` / `models_anthropic.py` — Pydantic models for request/response validation.
- `cache.py` — Thread-safe model metadata cache with TTL.
- `tokenizer.py` — tiktoken-based token counting with Claude correction factor.
- `utils.py` — ID generators (completion_id, message_id, tool_call_id), header builders.
- `exceptions.py` — Custom exception classes.

## Entry Point (main_copilot.py)

Mirrors `main.py` / `main_cursor.py`:
1. Load config from `.env`.
2. Validate credentials (GitHub token or SQLite).
3. Exchange for Copilot token.
4. Create shared HTTP client.
5. Fetch model list (with fallback).
6. Register OpenAI and Anthropic routers.
7. Start uvicorn on configured host:port (default 8002).
8. Startup banner with 🐙 emoji.

## Scope Exclusions

- No fake reasoning / extended thinking (tag injection).
- No truncation recovery.
- No debug middleware / debug logging.
- No protobuf encoding (upstream is plain JSON).
- No OAuth Device Flow (only SQLite + env var auth).
