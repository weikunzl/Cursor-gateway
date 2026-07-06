"""Shared pytest fixtures for Cursor Gateway tests."""

import pytest
from unittest.mock import AsyncMock, patch

import httpx


@pytest.fixture(scope="session", autouse=True)
def block_all_network_calls():
    """Block real httpx network calls in all tests."""
    mock_async_client = AsyncMock(spec=httpx.AsyncClient)

    async def network_call_error(*args, **kwargs):
        raise RuntimeError(
            "Real network request detected. Mock httpx.AsyncClient in tests."
        )

    mock_async_client.post.side_effect = network_call_error
    mock_async_client.get.side_effect = network_call_error
    mock_async_client.send.side_effect = network_call_error
    mock_async_client.stream.side_effect = network_call_error
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock()
    mock_async_client.aclose = AsyncMock()
    mock_async_client.is_closed = False

    patchers = [
        patch("cursor.auth.httpx.AsyncClient", return_value=mock_async_client),
        patch("cursor.http_client.httpx.AsyncClient", return_value=mock_async_client),
        patch("cursor.streaming_openai.httpx.AsyncClient", return_value=mock_async_client),
        patch("cursor.streaming_anthropic.httpx.AsyncClient", return_value=mock_async_client),
        patch("httpx.AsyncClient", return_value=mock_async_client),
    ]

    for patcher in patchers:
        patcher.start()

    yield

    for patcher in patchers:
        patcher.stop()
