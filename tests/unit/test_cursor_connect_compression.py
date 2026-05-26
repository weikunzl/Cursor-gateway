"""Tests for ConnectRPC compression-header negotiation against Cursor backend.

The Cursor backend (ConnectRPC server) rejects any envelope whose first-byte
``flags`` field has the compression bit set unless the request also carries
``Connect-Content-Encoding: <algo>`` so the server knows how to decompress.
Missing or wrong header → 200 OK with an in-stream error event:

    {"code":"internal","message":"received compressed envelope, but do not know how to decompress"}

That error silently kills Claude Code multi-turn loops (the second request is
always large enough to trip the gateway's auto-compression threshold).

These tests pin three layers of the fix together so the contract cannot
regress: the converter output, the header builder, and the HTTP client.
"""

from __future__ import annotations

import gzip
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from cursor.auth import CursorAuthManager
from cursor.converters_anthropic import anthropic_to_cursor
from cursor.converters_core import BuildResult
from cursor.converters_openai import build_cursor_payload as openai_build
from cursor.http_client import CursorHttpClient
from cursor.models_anthropic import AnthropicMessagesRequest
from cursor.models_openai import ChatCompletionRequest
from cursor.protobuf import wrap_connect_envelope
from cursor.utils import get_cursor_headers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auth() -> Mock:
    """Minimal CursorAuthManager stub for header building."""
    mgr = Mock(spec=CursorAuthManager)
    mgr.get_access_token = Mock(return_value="test-bearer-token")
    mgr.session_id = "session-uuid"
    mgr.client_key = "client-key-sha"
    mgr.machine_id = "machine-id-abcd"
    return mgr


def _make_user_msg(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def _three_turn_anthropic_request(model: str = "composer-2.5") -> AnthropicMessagesRequest:
    """Build a 3-message request — large enough to cross the compression threshold."""
    return AnthropicMessagesRequest(
        model=model,
        max_tokens=128,
        stream=True,
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "Grep",
                        "input": {"pattern": "mcp"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "no match",
                    }
                ],
            },
        ],
    )


def _three_turn_openai_request(model: str = "composer-2.5") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        stream=True,
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Grep", "arguments": '{"pattern":"mcp"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "no match"},
        ],
    )


# ---------------------------------------------------------------------------
# Envelope wrapper still works on either side of the fix.
# ---------------------------------------------------------------------------


class TestWrapConnectEnvelope:
    def test_uncompressed_envelope_flag_is_zero(self) -> None:
        env = wrap_connect_envelope(b"hello", compress=False)
        assert env[0] == 0x00
        assert env[5:] == b"hello"

    def test_compressed_envelope_flag_is_one_and_body_decompresses(self) -> None:
        env = wrap_connect_envelope(b"hello" * 100, compress=True)
        assert env[0] == 0x01
        assert gzip.decompress(env[5:]) == b"hello" * 100


# ---------------------------------------------------------------------------
# Header builder must surface the compression algorithm.
# ---------------------------------------------------------------------------


class TestGetCursorHeadersCompression:
    def test_no_content_encoding_by_default(self, auth: Mock) -> None:
        headers = get_cursor_headers(auth)
        assert "Connect-Content-Encoding" not in headers
        assert "Content-Encoding" not in headers

    def test_content_encoding_gzip_is_advertised(self, auth: Mock) -> None:
        headers = get_cursor_headers(auth, content_encoding="gzip")
        assert headers["Connect-Content-Encoding"] == "gzip"

    def test_content_encoding_identity_is_omitted(self, auth: Mock) -> None:
        """``identity`` means "no encoding" — same as not advertising at all."""
        headers = get_cursor_headers(auth, content_encoding="identity")
        assert "Connect-Content-Encoding" not in headers

    def test_content_encoding_none_is_omitted(self, auth: Mock) -> None:
        headers = get_cursor_headers(auth, content_encoding=None)
        assert "Connect-Content-Encoding" not in headers

    def test_accept_encoding_is_always_advertised(self, auth: Mock) -> None:
        """We must allow the server to send compressed responses back too."""
        headers = get_cursor_headers(auth)
        assert "gzip" in headers.get("Connect-Accept-Encoding", "")


# ---------------------------------------------------------------------------
# Converter must report whether the envelope is compressed so callers can
# set the header accordingly.
# ---------------------------------------------------------------------------


class TestBuildResultReportsCompression:
    def test_buildresult_has_compressed_field(self) -> None:
        result = BuildResult(
            payload=b"\x00\x00\x00\x00\x00",
            model_id="composer-2.5",
            message_count=1,
            compressed=False,
        )
        assert result.compressed is False

    def test_anthropic_converter_returns_buildresult(self) -> None:
        req = _three_turn_anthropic_request()
        result = anthropic_to_cursor(req, conversation_id="conv-1")
        assert isinstance(result, BuildResult)
        # 3 user/assistant turns → core_build_cursor_payload compresses
        assert result.compressed is True
        # Compressed envelope flag must match
        assert result.payload[0] == 0x01

    def test_anthropic_converter_uncompressed_for_short_history(self) -> None:
        req = AnthropicMessagesRequest(
            model="composer-2.5",
            max_tokens=64,
            stream=False,
            messages=[{"role": "user", "content": "hi"}],
        )
        result = anthropic_to_cursor(req, conversation_id="conv-1")
        assert isinstance(result, BuildResult)
        assert result.compressed is False
        assert result.payload[0] == 0x00

    def test_openai_converter_returns_buildresult(self) -> None:
        req = _three_turn_openai_request()
        result = openai_build(req, conversation_id="conv-2")
        assert isinstance(result, BuildResult)
        assert result.compressed is True
        assert result.payload[0] == 0x01


# ---------------------------------------------------------------------------
# HTTP client must inject the Connect-Content-Encoding header when (and only
# when) it is told the body is compressed.
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Stand-in for httpx.Response with a fixed status."""

    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class _FakeStreamClient:
    """Capture all outgoing requests through build_request / send / request."""

    def __init__(self) -> None:
        self.is_closed = False
        self.calls: List[Dict[str, Any]] = []

    def build_request(self, method: str, url: str, content: bytes = b"", headers=None):
        captured = {"method": method, "url": url, "content": content, "headers": dict(headers or {})}
        self.calls.append(captured)
        return SimpleNamespace(method=method, url=url, content=content, headers=dict(headers or {}))

    async def send(self, req, stream: bool = False) -> _FakeResponse:  # noqa: ARG002
        return _FakeResponse(200)

    async def request(self, method: str, url: str, content: bytes = b"", headers=None) -> _FakeResponse:
        self.calls.append({
            "method": method,
            "url": url,
            "content": content,
            "headers": dict(headers or {}),
        })
        return _FakeResponse(200)


@pytest.mark.asyncio
async def test_http_client_sets_gzip_header_when_compressed(auth: Mock) -> None:
    fake = _FakeStreamClient()
    client = CursorHttpClient(auth_manager=auth, shared_client=fake)

    await client.request_with_retry(
        "POST", "https://example.test/rpc",
        data=b"\x01\x00\x00\x00\x05hello",
        stream=True,
        compressed=True,
    )

    assert fake.calls, "expected at least one outbound request"
    headers = fake.calls[-1]["headers"]
    assert headers.get("Connect-Content-Encoding") == "gzip"


@pytest.mark.asyncio
async def test_http_client_omits_gzip_header_when_uncompressed(auth: Mock) -> None:
    fake = _FakeStreamClient()
    client = CursorHttpClient(auth_manager=auth, shared_client=fake)

    await client.request_with_retry(
        "POST", "https://example.test/rpc",
        data=b"\x00\x00\x00\x00\x05hello",
        stream=True,
        compressed=False,
    )

    headers = fake.calls[-1]["headers"]
    assert "Connect-Content-Encoding" not in headers


@pytest.mark.asyncio
async def test_http_client_unary_path_also_propagates_compression(auth: Mock) -> None:
    """Non-streaming requests use a different code path; both must set it."""
    fake = _FakeStreamClient()
    client = CursorHttpClient(auth_manager=auth, shared_client=fake)

    await client.request_with_retry(
        "POST", "https://example.test/rpc",
        data=b"\x01\x00\x00\x00\x05hello",
        stream=False,
        compressed=True,
    )

    headers = fake.calls[-1]["headers"]
    assert headers.get("Connect-Content-Encoding") == "gzip"


@pytest.mark.asyncio
async def test_http_client_default_keeps_uncompressed_header_absent(auth: Mock) -> None:
    """Backwards-compat: default ``compressed`` is False, header must stay absent."""
    fake = _FakeStreamClient()
    client = CursorHttpClient(auth_manager=auth, shared_client=fake)

    await client.request_with_retry(
        "POST", "https://example.test/rpc",
        data=b"\x00\x00\x00\x00\x05hello",
        stream=True,
        # no compressed kwarg
    )

    headers = fake.calls[-1]["headers"]
    assert "Connect-Content-Encoding" not in headers
