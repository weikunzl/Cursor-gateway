# Agent Reference: Architecture

This file expands the architecture notes summarized in `AGENTS.md`.

## System Role

Kiro Gateway is a Python FastAPI proxy server that exposes OpenAI-compatible and Anthropic-compatible APIs for Kiro, Amazon Q Developer, and AWS CodeWhisperer.

The gateway translates request and response formats, manages authentication, resolves model names, handles streaming, retries transient upstream failures, and classifies network errors.

## Layered Design

1. Routes layer: `routes_openai.py`, `routes_anthropic.py`
   - FastAPI endpoints, API-key checks, request validation, response shaping.
2. Converter layer: `converters_core.py`, `converters_openai.py`, `converters_anthropic.py`
   - OpenAI/Anthropic inputs to the internal Kiro payload shape.
3. Streaming layer: `streaming_core.py`, `streaming_openai.py`, `streaming_anthropic.py`
   - AWS event stream parsing and SSE output conversion.
4. Core services:
   - `auth.py`: token lifecycle and auth source detection.
   - `http_client.py`: retrying HTTP client and upstream calls.
   - `model_resolver.py`: model-name normalization and model lookup.
   - `cache.py`: model metadata cache.
5. Parsers:
   - `parsers.py`: AWS SSE/event-stream parsing.
   - `thinking_parser.py`: extended-thinking extraction.
6. Models:
   - `models_openai.py`, `models_anthropic.py`: Pydantic request/response schemas.

## Key Components

### Authentication

`KiroAuthManager` manages access-token lifecycle and refresh. It supports:

- Kiro IDE JSON credentials files.
- Direct refresh tokens from environment variables.
- `kiro-cli` SQLite credential databases.
- AWS SSO OIDC credentials for Builder ID and Enterprise flows.

Auth type is detected from available credentials. Token refresh must remain thread-safe with `asyncio.Lock` and should refresh before expiration.

### Model Resolution

Model resolution follows this pipeline:

1. Normalize external model names into Kiro-style names.
2. Check dynamically fetched models from the ListAvailableModels API.
3. Check manually configured hidden models.
4. Pass unknown models through to Kiro.

The gateway is not a model gatekeeper. Kiro API remains the final arbiter for unknown model names.

### HTTP Client

`KiroHttpClient` handles non-streaming upstream calls with retry behavior:

- `403`: refresh token and retry.
- `429`: exponential backoff.
- `5xx`: exponential backoff.
- timeout/network errors: classify and return actionable messages.

Use shared clients for non-streaming requests. Use per-request clients for streaming.

### Streaming

Streaming code converts Kiro/AWS event stream chunks into OpenAI or Anthropic SSE output. It must handle:

- first-token timeout and retry behavior.
- thinking blocks.
- tool-call parsing and deduplication.
- incomplete or malformed upstream chunks.

### Converters

`converters_core.py` owns shared request transformation logic:

- unified message format.
- tool processing and sanitization.
- message merging.
- Kiro payload construction.

OpenAI and Anthropic converter modules should stay thin adapters over the shared core.

## Important Patterns

### Per-Request Clients for Streaming

Always create an `httpx.AsyncClient` per streaming request to avoid CLOSE_WAIT leaks:

```python
async with httpx.AsyncClient(timeout=timeout) as client:
    async with client.stream("POST", url, json=payload) as response:
        async for line in response.aiter_lines():
            yield line
```

### Model Name Normalization

Examples:

- `claude-haiku-4-5-20251001` becomes `claude-haiku-4.5`.
- date suffixes are stripped when appropriate.
- dash minor versions are converted to dot minor versions.

### Tool Call Parsing

Kiro may return tool calls in bracketed JSON-like text instead of clean JSON. Parsers should tolerate known upstream quirks without modifying user intent.

### Thinking Block Extraction

Extended thinking uses a finite-state parser that separates thinking content from visible assistant text while preserving surrounding response content.

### Network Error Classification

Classify common user-facing failures:

- connect timeout.
- read timeout.
- DNS failure.
- proxy failure.
- upstream connection reset.

Messages should explain what happened and what the user can try next.
