# AGENTS.md - Cursor Gateway Agent Guide

This is the short, always-read guide for agents working in Cursor Gateway. Keep it concise; put expanded guidance in `docs/agents/` and link it from here.

## Project Context

Cursor Gateway is a Python 3.10+ FastAPI proxy that provides OpenAI-compatible and Anthropic-compatible APIs for the Cursor API.

It translates request/response formats, handles authentication from Cursor's local database, streaming, model resolution, retries, and user-friendly error classification.

Core identity: the gateway is a transparent proxy with minimal, purposeful modifications.

## Reference Documents

- Architecture details: `docs/agents/architecture.md`
- Development standards: `docs/agents/development.md`
- Operations and configuration: `docs/agents/operations.md`
- Troubleshooting, security, and performance: `docs/agents/troubleshooting-security.md`
- User documentation: `README.md`
- Test documentation: `tests/README.md`
- Environment template: `.env.example`

Read the relevant reference before changing related behavior.

## Core Principles

1. Preserve user intent and request structure.
2. Fix API-level quirks, not user decisions.
3. Make request changes only when required for validation, compatibility, or opt-in features.
4. Keep optional enhancements configurable and disableable.
5. Separate responsibilities: gateway handles API issues, clients handle content choices, models handle capacity limits.
6. Prefer systems that handle classes of issues over one-off patches.
7. Treat documentation, errors, and debug logs as part of the user experience.

## Hard Boundaries

The gateway may:

- fix API validation quirks.
- fix format incompatibilities.
- support authentication flows.
- add opt-in compatibility features.

The gateway must not:

- remove or rewrite user content unless explicitly required for API compatibility.
- decide which messages are important.
- trim context as a product choice.
- log secrets or raw credentials.
- hide behavior behind undocumented defaults.

## Project Structure

```text
cursor-gateway/
├── main.py
├── cursor/
│   ├── auth.py
│   ├── cache.py
│   ├── config.py
│   ├── converters_core.py
│   ├── converters_openai.py
│   ├── converters_anthropic.py
│   ├── http_client.py
│   ├── model_resolver.py
│   ├── models_openai.py
│   ├── models_anthropic.py
│   ├── parsers.py
│   ├── routes_openai.py
│   ├── routes_anthropic.py
│   ├── streaming_core.py
│   ├── streaming_openai.py
│   ├── streaming_anthropic.py
│   ├── thinking_split.py
│   ├── bracket_tools.py
│   ├── yaml_tools.py
│   └── tool_args.py
├── tests/
│   ├── conftest.py
│   └── unit/
├── docs/agents/
├── scripts/
├── .env.example
├── requirements.txt
└── pytest.ini
```

## Architecture Summary

The codebase is layered:

1. Routes: FastAPI endpoints, auth checks, request validation.
2. Converters: OpenAI/Anthropic requests to Cursor ConnectRPC payloads.
3. Streaming: Cursor event streams to OpenAI/Anthropic SSE.
4. Core services: auth, HTTP client, model resolution, caching.
5. Parsers: stream parsing, thinking blocks, tool-call extraction.
6. Models: Pydantic schemas.

Important invariants:

- use per-request `httpx.AsyncClient` instances for streaming.
- use shared clients only for non-streaming connection pooling.
- keep OpenAI and Anthropic adapters thin over shared converter logic.
- pass unknown model names through after alias/normalization checks.

See `docs/agents/architecture.md`.

## Code Standards

- Use type hints for every function parameter and return value.
- Use Google-style docstrings for public functions and non-trivial helpers.
- Use loguru for logging.
- Use async I/O for request-path network and file operations.
- Avoid bare `except:` and avoid broad exception handling without context.
- Keep user-facing errors actionable and sanitized.

See `docs/agents/development.md`.

## Testing Standards

All behavior changes require tests. Network isolation is mandatory; `tests/conftest.py` blocks real httpx calls.

```bash
pytest tests/unit/test_<module>.py -v
pytest -v
```

## Operations Summary

```bash
pip install -r requirements.txt
python main.py
./scripts/cursor-gateway.sh start
docker-compose up -d
```

Core endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/messages`.

See `docs/agents/operations.md`.

## Collaboration Workflow

1. Read relevant code and reference docs first.
2. Keep edits scoped to the requested behavior.
3. Add or update tests for behavior changes.
4. Run focused tests, then broader tests when shared behavior changed.
5. Do not commit or push unless explicitly asked.

## Git Notes

Follow existing commit style:

```text
<type>(<scope>): <description>
```
