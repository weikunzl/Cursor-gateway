# Agent Reference: Development Standards

This file expands the coding, testing, and contribution standards summarized in `AGENTS.md`.

## Code Style

- Use Python 3.10+ features that improve clarity without reducing readability.
- Use `snake_case` for functions and variables.
- Use `PascalCase` for classes.
- Use `UPPER_SNAKE_CASE` for constants.
- Use `_leading_underscore` for private helpers and attributes.
- Keep modules focused on one responsibility.
- Prefer existing local helpers and patterns over new abstractions.

## Type Hints

Every function parameter and return value must be typed.

```python
def extract_text_content(content: Any) -> str:
    """Extract text from supported content formats."""
    ...

async def refresh_token(self) -> str:
    """Refresh and return an access token."""
    ...
```

## Docstrings

Use Google-style docstrings for public functions, classes, and non-trivial private helpers.

Include `Args`, `Returns`, and `Raises` when applicable. Examples are useful when behavior is subtle.

```python
def normalize_model_name(name: str) -> str:
    """
    Normalize a client model name to the Kiro model format.

    Args:
        name: External model name from a client.

    Returns:
        Normalized model name for Kiro API calls.
    """
    ...
```

## Logging

Use loguru.

- `DEBUG`: detailed diagnostic information.
- `INFO`: important business decisions and lifecycle events.
- `WARNING`: recoverable or unusual behavior.
- `ERROR`: failed operations that need attention.

Never log credentials, access tokens, refresh tokens, API keys, passwords, or raw secrets.

## Error Handling

- Do not use bare `except:`.
- Avoid broad `except Exception:` unless there is no narrower useful exception and the handler adds context before re-raising or returning.
- API errors should be actionable and user-friendly.
- Internal details should be logged safely, not exposed to clients.

```python
try:
    result = await some_operation()
except httpx.TimeoutException as exc:
    logger.warning(f"Upstream request timed out: {exc}")
    raise HTTPException(status_code=504, detail="Kiro API timed out. Please retry.") from exc
```

## Async I/O

All network and file I/O on request paths should be async where the surrounding framework supports it.

Use shared clients for non-streaming work and per-request clients for streaming work.

## Testing Rules

All behavior changes require tests.

Tests should try to break the code:

- happy paths.
- malformed inputs.
- boundary values.
- upstream quirks.
- error paths.
- retry behavior.
- compatibility behavior.

## Network Isolation

Tests must not make real network calls.

`tests/conftest.py` includes global network blocking for httpx. Mock all external services explicitly.

## Test Structure

Use Arrange-Act-Assert.

```python
@pytest.mark.asyncio
async def test_token_refresh_success(mock_env_vars, mock_kiro_token_response):
    """Verify successful token refresh updates the cached token."""
    # Arrange
    auth_manager = KiroAuthManager()

    # Act
    token = await auth_manager.get_valid_token()

    # Assert
    assert token
```

## Test Organization

- Unit tests live under `tests/unit/`.
- Integration tests live under `tests/integration/`.
- Prefer class names such as `TestAuthSuccess`, `TestAuthErrors`, `TestAuthEdgeCases`.
- Prefer test names like `test_<condition>_<expected_result>`.

## Required Commands

Run the narrowest meaningful tests first, then broaden when risk warrants it.

```bash
pytest tests/unit/test_<module>.py -v
pytest -v
pytest --cov=kiro --cov-report=html
```

## Git Workflow

Before committing, inspect status, diff, and recent commit style.

Commit messages generally follow:

```text
<type>(<scope>): <description>
```

Examples:

- `feat(auth): add support for new auth method`
- `fix(streaming): handle empty chunks correctly`
- `docs: update configuration examples`

Do not commit secrets. Do not push unless explicitly asked.

## Common Development Tasks

### Add an Endpoint

1. Define or update Pydantic models.
2. Add the route.
3. Add converter behavior.
4. Add streaming behavior if needed.
5. Add focused tests.

### Add a Hidden Model

Add hidden models in `kiro/config.py` only when dynamic resolution cannot discover them.

### Make a Behavioral Change

1. Read the relevant modules first.
2. Add tests that describe the intended behavior and edge cases.
3. Implement the smallest coherent change.
4. Run focused tests.
5. Run broader tests when shared behavior changed.
