# Agent Reference: Architecture

Cursor Gateway exposes OpenAI-compatible and Anthropic-compatible APIs over the Cursor ConnectRPC backend.

## Layered Design

1. Routes: `routes_openai.py`, `routes_anthropic.py`
2. Converters: `converters_core.py`, `converters_openai.py`, `converters_anthropic.py`
3. Streaming: `streaming_core.py`, `streaming_openai.py`, `streaming_anthropic.py`
4. Core services: `auth.py`, `http_client.py`, `model_resolver.py`, `cache.py`
5. Tool/thinking helpers: `bracket_tools.py`, `yaml_tools.py`, `thinking_split.py`, `tool_args.py`
6. Models: `models_openai.py`, `models_anthropic.py`

## Authentication

`CursorAuthManager` reads credentials from Cursor's `state.vscdb` SQLite database or from `CURSOR_ACCESS_TOKEN` / `CURSOR_MACHINE_ID` environment variables.

## Model Resolution

1. Apply configured aliases (e.g. Claude Code model IDs to `composer-2.5`).
2. Check cached models from Cursor's AvailableModels RPC.
3. Pass unknown models through to Cursor.

## Streaming

- Use per-request `httpx.AsyncClient` for streaming.
- Parse ConnectRPC/protobuf event streams.
- Convert to OpenAI or Anthropic SSE output.
- Handle thinking blocks and inline tool-call text dialects.

## Tool Call Extraction

Cursor models may emit tool calls as:

- structured tool blocks.
- bracket text: `[Tool Call: Name({...})]`.
- YAML or ASCII markup in composer output.

Parsers should preserve user-visible text and only structure known tool dialects.
