"""
Parse Cursor composer redacted tool-call text into structured tool invocations.

Cursor sometimes embeds tool intent as plain text:

    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
    Grep
    <｜tool▁sep｜>pattern
    superpowers
    <｜tool▁call▁end｜>...

Claude Code expects Anthropic ``tool_use`` blocks instead.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Build with ASCII pipe (0x7C) — avoid Unicode lookalikes in source literals
_PIPE = chr(124)
TOOL_CALLS_BEGIN = f"<{_PIPE}redacted_tool_calls_begin{_PIPE}>"
TOOL_CALLS_END = f"<{_PIPE}redacted_tool_calls_end{_PIPE}>"
TOOL_CALL_BEGIN = f"<{_PIPE}redacted_tool_call_begin{_PIPE}>"
TOOL_CALL_END = f"<{_PIPE}redacted_tool_call_end{_PIPE}>"
TOOL_ARG_SEP = f"<{_PIPE}redacted_tool_sep{_PIPE}>"

# Longest first for suffix holdback when a marker may be split across chunks
_MARKER_PREFIXES: Tuple[str, ...] = tuple(
    sorted(
        {TOOL_CALLS_BEGIN, TOOL_CALLS_END, TOOL_CALL_BEGIN, TOOL_CALL_END, TOOL_ARG_SEP},
        key=len,
        reverse=True,
    )
)


@dataclass
class ParsedToolCall:
    """A single tool call extracted from redacted text."""

    name: str
    arguments: Dict[str, Any]


def _holdback_suffix(data: str) -> Tuple[str, str]:
    """
    Split data into safe-to-emit prefix and suffix that may complete a marker.

    Args:
        data: Buffered text not yet emitted

    Returns:
        (safe_prefix, held_suffix)
    """
    if not data:
        return "", ""

    max_hold = 0
    for marker in _MARKER_PREFIXES:
        for prefix_len in range(1, min(len(marker), len(data)) + 1):
            if data.endswith(marker[:prefix_len]):
                max_hold = max(max_hold, prefix_len)

    if max_hold == 0:
        return data, ""
    return data[:-max_hold], data[-max_hold:]


def _parse_single_tool_call(body: str) -> ParsedToolCall:
    """
    Parse one redacted tool call body into name and arguments.

    Format::

        ToolName
        <｜tool▁sep｜>arg1
        value1
        <｜tool▁sep｜>arg2
        value2

    Args:
        body: Text between tool_call_begin and tool_call_end markers

    Returns:
        ParsedToolCall with tool name and argument dict
    """
    stripped = body.strip()
    if not stripped:
        return ParsedToolCall(name="", arguments={})

    if TOOL_ARG_SEP not in stripped:
        return ParsedToolCall(name=stripped.split("\n", 1)[0].strip(), arguments={})

    segments = stripped.split(TOOL_ARG_SEP)
    name = segments[0].strip()
    arguments: Dict[str, Any] = {}

    for segment in segments[1:]:
        segment = segment.strip()
        if not segment:
            continue
        key, _, value = segment.partition("\n")
        key = key.strip()
        if key:
            arguments[key] = value.strip()

    return ParsedToolCall(name=name, arguments=arguments)


def _parse_tool_calls_block(block: str) -> List[ParsedToolCall]:
    """Parse all tool calls inside a redacted_tool_calls block."""
    tools: List[ParsedToolCall] = []
    search_from = 0

    while True:
        begin_idx = block.find(TOOL_CALL_BEGIN, search_from)
        if begin_idx == -1:
            break

        body_start = begin_idx + len(TOOL_CALL_BEGIN)
        end_idx = block.find(TOOL_CALL_END, body_start)
        if end_idx == -1:
            break

        body = block[body_start:end_idx]
        parsed = _parse_single_tool_call(body)
        if parsed.name:
            tools.append(parsed)

        search_from = end_idx + len(TOOL_CALL_END)

    return tools


def extract_redacted_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Remove redacted tool blocks from text and return structured tool calls.

    Args:
        text: Assistant visible text that may contain redacted tool markup

    Returns:
        Tuple of (cleaned_text, tool_dicts) where each tool dict has name and arguments
    """
    if TOOL_CALLS_BEGIN not in text:
        return text, []

    cleaned_parts: List[str] = []
    tools: List[Dict[str, Any]] = []
    remainder = text

    while TOOL_CALLS_BEGIN in remainder:
        begin_idx = remainder.find(TOOL_CALLS_BEGIN)
        if begin_idx > 0:
            cleaned_parts.append(remainder[:begin_idx])

        remainder = remainder[begin_idx + len(TOOL_CALLS_BEGIN) :]
        # Use rfind so nested markers cannot confuse the outer block boundary
        end_idx = remainder.rfind(TOOL_CALLS_END)
        if end_idx == -1:
            cleaned_parts.append(TOOL_CALLS_BEGIN + remainder)
            remainder = ""
            break

        block = remainder[:end_idx]
        remainder = remainder[end_idx + len(TOOL_CALLS_END) :]

        for parsed in _parse_tool_calls_block(block):
            tools.append({"name": parsed.name, "arguments": parsed.arguments})

    if remainder:
        cleaned_parts.append(remainder)

    cleaned = "".join(cleaned_parts)
    return cleaned, tools


class RedactedToolStreamProcessor:
    """
    Incrementally extract redacted tool calls from streamed visible text.

    Emits safe text prefixes immediately and buffers incomplete tool blocks until
    the closing marker arrives.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a visible-text chunk.

        Args:
            chunk: New visible text from the model stream

        Returns:
            Tuple of (text_to_emit_now, tool_calls_parsed_now)
        """
        if not chunk:
            return "", []

        self._buffer += chunk
        text_out: List[str] = []
        tools_out: List[Dict[str, Any]] = []

        while TOOL_CALLS_BEGIN in self._buffer:
            begin_idx = self._buffer.find(TOOL_CALLS_BEGIN)
            prefix = self._buffer[:begin_idx]
            if prefix:
                text_out.append(prefix)

            self._buffer = self._buffer[begin_idx:]
            end_idx = self._buffer.find(TOOL_CALLS_END)
            if end_idx == -1:
                break

            segment = self._buffer[: end_idx + len(TOOL_CALLS_END)]
            self._buffer = self._buffer[end_idx + len(TOOL_CALLS_END) :]

            _, parsed = extract_redacted_tool_calls(segment)
            tools_out.extend(parsed)

        if TOOL_CALLS_BEGIN not in self._buffer:
            safe, held = _holdback_suffix(self._buffer)
            if safe:
                text_out.append(safe)
            self._buffer = held

        return "".join(text_out), tools_out

    def flush(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Flush remaining buffer at end of stream.

        Incomplete tool markup is emitted as plain text rather than dropped.
        """
        if not self._buffer:
            return "", []

        remaining = self._buffer
        self._buffer = ""
        cleaned, tools = extract_redacted_tool_calls(remaining)
        return cleaned, tools
