# AGENTS.md - Kiro Gateway Agent Guide

This is the short, always-read guide for agents working in Kiro Gateway. Keep it concise; put expanded guidance in `docs/agents/` and link it from here.

## Project Context

Kiro Gateway is a Python 3.10+ FastAPI proxy that provides OpenAI-compatible and Anthropic-compatible APIs for Kiro, Amazon Q Developer, and AWS CodeWhisperer.

It translates request/response formats, handles authentication, streaming, model resolution, retries, debug logging, and user-friendly error classification.

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
kiro-gateway/
├── main.py
├── kiro/                 # routes, converters, streaming, auth, HTTP, models
├── tests/                # unit, integration, shared fixtures
├── docs/agents/          # expanded agent references
├── .env.example
├── requirements.txt
└── pytest.ini
```

## Architecture Summary

The codebase is layered:

1. Routes: FastAPI endpoints, auth checks, request validation.
2. Converters: OpenAI/Anthropic requests to Kiro payloads.
3. Streaming: Kiro/AWS event streams to OpenAI/Anthropic SSE.
4. Core services: auth, HTTP client, model resolution, caching.
5. Parsers: AWS stream parsing and thinking block extraction.
6. Models: Pydantic schemas.

Important invariants:

- use per-request `httpx.AsyncClient` instances for streaming.
- use shared clients only for non-streaming connection pooling.
- keep OpenAI and Anthropic adapters thin over shared converter logic.
- pass unknown model names through to Kiro after normalization/cache checks.

See `docs/agents/architecture.md`.

## Code Standards

- Use type hints for every function parameter and return value.
- Use Google-style docstrings for public functions and non-trivial helpers.
- Use loguru for logging.
- Use async I/O for request-path network and file operations.
- Use focused modules and existing local helpers before adding new abstractions.
- Avoid bare `except:` and avoid broad exception handling without context.
- Keep user-facing errors actionable and sanitized.

See `docs/agents/development.md`.

## Testing Standards

All behavior changes require tests. Tests should cover happy paths, malformed inputs, edge cases, upstream quirks, and error paths.

Network isolation is mandatory. Tests must mock external services; real network calls are blocked by `tests/conftest.py`.

Useful commands:

```bash
pytest tests/unit/test_<module>.py -v
pytest -v
pytest --cov=kiro --cov-report=html
```

## Operations Summary

Common commands:

```bash
pip install -r requirements.txt
python main.py
python main.py --port 9000
uvicorn main:app --host 0.0.0.0 --port 8000
docker-compose up -d
```

Core endpoints:

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/messages`

See `docs/agents/operations.md`.

## Security and Performance

- Never log credentials, tokens, API keys, passwords, or raw authorization headers.
- Treat debug logs as sensitive.
- Validate inputs with Pydantic.
- Use HTTPS in production.
- Stream large responses instead of buffering them.
- Cache model metadata where appropriate.
- Avoid unnecessary deep copies and JSON round-trips on hot paths.

See `docs/agents/troubleshooting-security.md`.

## Debugging Workflow

For vague upstream errors such as `Improperly formed request`:

1. Reproduce with the smallest failing request.
2. Enable `DEBUG_MODE="errors"` when useful.
3. Compare converter output with a passing request.
4. Identify the specific API-level incompatibility.
5. Add a regression test.
6. Fix only the compatibility issue.

Do not infer user intent from vague upstream validation errors.

## Collaboration Workflow

When making changes:

1. Read the relevant code and reference docs first.
2. Check existing patterns before designing new ones.
3. Keep edits scoped to the requested behavior.
4. Add or update tests for behavior changes.
5. Run focused tests, then broader tests when shared behavior changed.
6. Check for lint/type issues when available.
7. Do not commit or push unless explicitly asked.

If the working tree is dirty, preserve user changes. Never revert unrelated edits without explicit permission.

## Git Notes

Before any requested commit, inspect status, diff, and recent log. Follow the repository's existing commit style:

```text
<type>(<scope>): <description>
```
