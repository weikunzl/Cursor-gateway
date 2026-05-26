"""End-to-end tests for the Cursor → Anthropic streaming pipeline.

These tests pin the contract that DeepSeek-native tool-call tokens emitted by
Cursor as plain ``content`` events are converted into proper Anthropic
``tool_use`` SSE content blocks. Regressing this contract was the root cause
of the user-reported "raw tool token soup" bug.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import MagicMock

import pytest

from cursor import streaming_anthropic as streaming
from cursor.streaming_core import CursorEvent


PIPE = "\uff5c"  # ｜  FULLWIDTH VERTICAL LINE
UBAR = "\u2581"  # ▁  LOWER ONE EIGHTH BLOCK

CALLS_BEGIN = f"<{PIPE}tool{UBAR}calls{UBAR}begin{PIPE}>"
CALLS_END = f"<{PIPE}tool{UBAR}calls{UBAR}end{PIPE}>"
CALL_BEGIN = f"<{PIPE}tool{UBAR}call{UBAR}begin{PIPE}>"
CALL_END = f"<{PIPE}tool{UBAR}call{UBAR}end{PIPE}>"
SEP = f"<{PIPE}tool{UBAR}sep{PIPE}>"


def _parse_sse(events: List[str]) -> List[Dict[str, Any]]:
    """Decode SSE strings into a flat list of ``data:`` payloads."""
    parsed: List[Dict[str, Any]] = []
    for raw in events:
        for line in raw.splitlines():
            if line.startswith("data: "):
                parsed.append(json.loads(line[len("data: ") :]))
    return parsed


def _make_cursor_event_stream(chunks: List[str]) -> AsyncGenerator[CursorEvent, None]:
    """Build an async iterable of Cursor ``content`` events from text chunks."""

    async def _gen() -> AsyncGenerator[CursorEvent, None]:
        for chunk in chunks:
            yield CursorEvent(type="content", content=chunk)

    return _gen()


@pytest.fixture
def patched_parse(monkeypatch: pytest.MonkeyPatch):
    """Replace ``parse_cursor_stream`` with a programmable async generator."""

    def _install(chunks: List[str]) -> None:
        def _fake(_response, _timeout=None):  # noqa: ANN001
            return _make_cursor_event_stream(chunks)

        monkeypatch.setattr(streaming, "parse_cursor_stream", _fake)

    return _install


@pytest.fixture
def mock_response() -> MagicMock:
    response = MagicMock()

    async def _aclose() -> None:
        return None

    response.aclose = _aclose
    return response


@pytest.fixture
def mock_model_cache() -> MagicMock:
    return MagicMock()


@pytest.mark.asyncio
async def test_user_reported_skill_call_becomes_tool_use_block(
    patched_parse, mock_response, mock_model_cache
):
    """Regression test mirroring the user's failing 'Skill' invocation."""

    user_block = (
        "正在使用 ONES 相关技能查询您昨天创建的任务。\n\n"
        f"{CALLS_BEGIN}{CALL_BEGIN}\n"
        "Skill\n"
        f"{SEP}skill_name\n"
        "devops:my-requirements\n"
        f"{CALL_END}{CALLS_END}"
    )
    patched_parse([user_block])

    events: List[str] = []
    async for chunk in streaming.stream_cursor_to_anthropic(
        response=mock_response,
        model="deepseek-v3.2",
        model_cache=mock_model_cache,
        first_token_timeout=5.0,
        request_messages=[{"role": "user", "content": "查询一下我昨天创建的任务"}],
    ):
        events.append(chunk)

    payloads = _parse_sse(events)

    # Find tool_use block start
    tool_use_starts = [
        p for p in payloads
        if p.get("type") == "content_block_start"
        and p.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_use_starts) == 1
    assert tool_use_starts[0]["content_block"]["name"] == "Skill"

    # The JSON arguments delta carries the structured input
    input_deltas = [
        p for p in payloads
        if p.get("type") == "content_block_delta"
        and p.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert len(input_deltas) == 1
    assert json.loads(input_deltas[0]["delta"]["partial_json"]) == {
        "skill_name": "devops:my-requirements"
    }

    # The visible Chinese sentence is emitted as a text_delta with no leaked tokens
    text_deltas = [
        p for p in payloads
        if p.get("type") == "content_block_delta"
        and p.get("delta", {}).get("type") == "text_delta"
    ]
    full_text = "".join(d["delta"]["text"] for d in text_deltas)
    assert "正在使用 ONES 相关技能查询您昨天创建的任务" in full_text
    assert PIPE not in full_text
    assert UBAR not in full_text

    # The final message_delta must signal tool_use so the client invokes the tool
    message_deltas = [p for p in payloads if p.get("type") == "message_delta"]
    assert message_deltas, "expected a message_delta event"
    assert message_deltas[-1]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_tokens_split_across_content_events_still_parse(
    patched_parse, mock_response, mock_model_cache
):
    """Markers split across many Cursor content events must still convert."""

    full = (
        "prefix "
        f"{CALLS_BEGIN}{CALL_BEGIN}\n"
        "Echo\n"
        f"{SEP}text\n"
        "hello\n"
        f"{CALL_END}{CALLS_END}"
        " suffix"
    )
    # Chunk roughly every 3 characters to force splits inside markers
    chunks = [full[i : i + 3] for i in range(0, len(full), 3)]
    patched_parse(chunks)

    events: List[str] = []
    async for chunk in streaming.stream_cursor_to_anthropic(
        response=mock_response,
        model="deepseek-v3.2",
        model_cache=mock_model_cache,
        first_token_timeout=5.0,
    ):
        events.append(chunk)

    payloads = _parse_sse(events)

    tool_starts = [
        p for p in payloads
        if p.get("type") == "content_block_start"
        and p.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "Echo"

    text_deltas = [
        p for p in payloads
        if p.get("type") == "content_block_delta"
        and p.get("delta", {}).get("type") == "text_delta"
    ]
    visible = "".join(d["delta"]["text"] for d in text_deltas)
    assert visible.startswith("prefix ")
    assert "suffix" in visible
    assert PIPE not in visible
    assert UBAR not in visible
