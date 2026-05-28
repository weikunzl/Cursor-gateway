"""Tests for Cursor gateway model alias resolution."""

from __future__ import annotations

import pytest

from cursor.cache import ModelInfoCache
from cursor.config import MODEL_ALIASES
from cursor.model_resolver import ModelResolver


@pytest.fixture
def resolver() -> ModelResolver:
    """ModelResolver backed by an empty cache (aliases only)."""
    return ModelResolver(cache=ModelInfoCache(), aliases=MODEL_ALIASES)


@pytest.mark.parametrize(
    "client_model",
    [
        "claude-opus-4-7",
        "claude-opus-4-7-thinking-xhign",
        "claude-opus-4-7-thinking-high",
        "claude-opus-4-6",
    ],
)
def test_claude_code_models_resolve_to_composer(
    resolver: ModelResolver,
    client_model: str,
) -> None:
    """Claude Code default model IDs must map to a working Cursor model."""
    assert resolver.resolve(client_model) == "composer-2.5"
