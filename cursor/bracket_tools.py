"""
Parse the ``[Tool Call: Name({json_args})]`` text dialect that Cursor's
``composer-2.5`` model sometimes emits when it bypasses structured tool
calling and narrates the invocation inline instead.

Real-world example (captured from ``composer-2.5`` via cursor-gateway)::

    继续扩大搜索范围，查找 MCP 相关配置与文档。

    [Tool Call: Grep({"pattern": "mcp", "head_limit": "80"})]
    [Tool Call: Glob({"glob_pattern": "**/*mcp*"})]

Claude Code only executes tools delivered as proper Anthropic ``tool_use``
content blocks, so the gateway intercepts these inline calls and converts
them to structured form before they reach the client. The parser is paired
with :class:`cursor.redacted_tools.RedactedToolStreamProcessor` (which
handles the DeepSeek-native ``<｜tool▁call▁begin｜>`` dialect); together
they cover every tool-call text format we've observed from composer-2.5.

Implementation notes:

* Only the literal ``"[Tool Call: "`` prefix is matched, so ordinary
  Markdown links (``[label](url)``) are not affected.
* The closing ``)]`` is found by tracking balanced parentheses *and*
  JSON braces while honouring string literals — argument values may
  contain any of ``( ) [ ] { }``.
* Argument bodies are parsed strictly as JSON objects (per the format
  emitted by composer-2.5). A non-JSON or non-object body is treated as
  malformed and the original text is preserved verbatim.
* Streaming: a partial ``[Tool Call: `` prefix at the end of a chunk is
  held back so a marker split across chunks is still recognised once the
  next chunk arrives.
* Unterminated or malformed calls are emitted as plain text — we never
  silently swallow visible content.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

PREFIX = "[Tool Call: "
SUFFIX = ")]"


def _is_identifier_char(ch: str) -> bool:
    """Tool names are ``[A-Za-z0-9_]+`` — strict on purpose."""
    return ch.isalnum() or ch == "_"


def _scan_balanced_call(buf: str, start: int) -> Optional[Tuple[str, str, int]]:
    """
    Try to read one ``Name(JSON_OBJECT)]`` starting at ``buf[start]``.

    Args:
        buf: Buffer containing the unconsumed text starting with ``PREFIX``.
        start: Index inside ``buf`` of the first character after ``PREFIX``
            (i.e. the start of the tool name).

    Returns:
        Tuple ``(tool_name, json_args_str, end_index)`` on success, where
        ``end_index`` is the position just past the closing ``)]``.
        Returns ``None`` if the buffer is incomplete (more chunks needed)
        or malformed.
    """
    i = start
    name_start = i
    while i < len(buf) and _is_identifier_char(buf[i]):
        i += 1
    name = buf[name_start:i]
    if not name:
        return None
    if i >= len(buf):
        # Need more data to see the ``(``.
        return None
    if buf[i] != "(":
        # Not our shape — caller will emit ``[Tool Call: `` verbatim.
        return None

    # Walk the argument body, balancing parens/braces and respecting strings.
    i += 1  # skip the opening '('
    json_start = i
    depth_paren = 1
    depth_brace = 0
    in_string = False
    escape_next = False

    while i < len(buf):
        ch = buf[i]
        if escape_next:
            escape_next = False
        elif in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren -= 1
                if depth_paren == 0:
                    break
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1
        i += 1

    if i >= len(buf) or depth_paren != 0:
        # Incomplete: closing ')' not seen yet.
        return None

    json_str = buf[json_start:i]
    i += 1  # skip the closing ')'

    if i >= len(buf):
        # Need the trailing ']' too.
        return None
    if buf[i] != "]":
        return None
    i += 1  # skip ']'

    return name, json_str, i


def _parse_json_args(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the body inside ``Name(...)``. Must be a JSON object."""
    stripped = raw.strip()
    if not stripped:
        # ``Name()`` shape — accept as empty args.
        return {}
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _holdback_prefix_suffix(data: str) -> Tuple[str, str]:
    """
    Hold back the longest suffix of ``data`` that could grow into ``PREFIX``.

    Args:
        data: Buffered text not yet emitted downstream.

    Returns:
        Tuple ``(safe_prefix, held_suffix)``. ``held_suffix`` is empty when
        no trailing characters could conceivably complete the marker.
    """
    if not data:
        return "", ""

    max_hold = 0
    for prefix_len in range(1, min(len(PREFIX), len(data)) + 1):
        if data.endswith(PREFIX[:prefix_len]):
            if prefix_len > max_hold:
                max_hold = prefix_len

    if max_hold == 0:
        return data, ""
    return data[:-max_hold], data[-max_hold:]


def _consume_one(buf: str) -> Tuple[Optional[Dict[str, Any]], str, int]:
    """
    Try to consume a single ``[Tool Call: ...)]`` starting at ``buf[0]``.

    Args:
        buf: Buffer whose first character is ``"["`` of a ``PREFIX`` match.

    Returns:
        ``(tool_dict_or_None, leftover_buf, consumed_len)``:
        - ``tool_dict_or_None`` is ``None`` when the body is malformed or
          incomplete; the caller decides whether to emit the raw text or
          wait for more data.
        - ``consumed_len`` is how many chars of ``buf`` were consumed when
          a tool is returned. ``0`` when the caller needs more data.
    """
    if not buf.startswith(PREFIX):
        return None, buf, 0

    scan = _scan_balanced_call(buf, len(PREFIX))
    if scan is None:
        # Incomplete — caller must wait for more data.
        return None, buf, 0

    name, json_str, end = scan
    args = _parse_json_args(json_str)
    if args is None:
        # Malformed JSON: return the raw call verbatim, advance past it
        # by sentinel value to signal "skip but keep text".
        return None, buf, -1
    return {"name": name, "arguments": args}, buf[end:], end


def extract_bracket_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Remove every ``[Tool Call: Name({json})]`` from ``text``.

    Args:
        text: Text that may contain zero or more bracket-format tool calls.

    Returns:
        Tuple ``(cleaned_text, tool_dicts)``. Each tool dict has the shape
        ``{"name": str, "arguments": dict}``. Unterminated or malformed
        calls are preserved verbatim in ``cleaned_text``.
    """
    if PREFIX not in text:
        return text, []

    cleaned_parts: List[str] = []
    tools: List[Dict[str, Any]] = []
    remainder = text

    while True:
        idx = remainder.find(PREFIX)
        if idx == -1:
            cleaned_parts.append(remainder)
            break

        if idx > 0:
            cleaned_parts.append(remainder[:idx])

        tail = remainder[idx:]
        scan = _scan_balanced_call(tail, len(PREFIX))
        if scan is None:
            # Unterminated — keep verbatim and stop.
            cleaned_parts.append(tail)
            remainder = ""
            break

        name, json_str, end = scan
        args = _parse_json_args(json_str)
        if args is None:
            # Malformed JSON — keep this call verbatim and continue past it.
            cleaned_parts.append(tail[:end])
            remainder = tail[end:]
            continue

        tools.append({"name": name, "arguments": args})
        remainder = tail[end:]

    return "".join(cleaned_parts), tools


class BracketToolCallProcessor:
    """
    Incrementally extract ``[Tool Call: Name({...})]`` calls from streamed
    visible text.

    Mirrors the public API of
    :class:`cursor.redacted_tools.RedactedToolStreamProcessor` so the two
    can be chained inside the streaming layer.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a streamed text chunk.

        Args:
            chunk: New visible text from the model stream.

        Returns:
            Tuple ``(text_to_emit_now, tool_calls_parsed_now)``. Either
            side may be empty when the chunk only contained partial
            markers or partial tool bodies.
        """
        if not chunk:
            return "", []

        self._buffer += chunk
        text_out: List[str] = []
        tools_out: List[Dict[str, Any]] = []

        while True:
            idx = self._buffer.find(PREFIX)
            if idx == -1:
                # No complete marker in buffer; emit safe-to-flush prefix.
                safe, held = _holdback_prefix_suffix(self._buffer)
                if safe:
                    text_out.append(safe)
                self._buffer = held
                break

            # Emit any plain text before the marker.
            if idx > 0:
                text_out.append(self._buffer[:idx])
                self._buffer = self._buffer[idx:]

            scan = _scan_balanced_call(self._buffer, len(PREFIX))
            if scan is None:
                # Incomplete call — wait for more data.
                break

            name, json_str, end = scan
            args = _parse_json_args(json_str)
            if args is None:
                # Malformed: drop into visible text verbatim and skip past.
                text_out.append(self._buffer[:end])
                self._buffer = self._buffer[end:]
                continue

            tools_out.append({"name": name, "arguments": args})
            self._buffer = self._buffer[end:]

        return "".join(text_out), tools_out

    def flush(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Drain any remaining buffer at end of stream.

        Unterminated tool markup is emitted as plain text rather than
        silently dropped, so the user still sees what the model said.
        """
        if not self._buffer:
            return "", []

        remaining = self._buffer
        self._buffer = ""
        cleaned, tools = extract_bracket_tool_calls(remaining)
        return cleaned, tools
