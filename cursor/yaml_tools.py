"""
Parse the YAML-style ``[Tool call: Name`` inline tool dialect.

composer-2.5 sometimes emits tool invocations as plain text instead of
structured protobuf tool_use fields::

    [Tool call: Grep
      pattern: foo
      path: /tmp
      head_limit: 50

This is distinct from the JSON bracket dialect handled by
``cursor.bracket_tools`` (``[Tool Call: Name({...})]``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from cursor.tool_args import normalize_tool_call

# ``[Tool call: Grep`` or ``[Tool call] Grep`` — case-sensitive on purpose.
_YAML_TOOL_START = re.compile(r"\[Tool call(?::|\])\s*(\w+)\s*\n")

# Indented ``key: value`` lines following a tool header.
_YAML_ARG_LINE = re.compile(r"^(\s{2,})(\w+):\s*(.*)$")

# Hold back partial ``[Tool call`` prefixes at chunk boundaries.
_YAML_PREFIX = "[Tool call"


def _holdback_yaml_prefix(data: str) -> Tuple[str, str]:
    if not data:
        return "", ""
    max_hold = 0
    for prefix_len in range(1, min(len(_YAML_PREFIX), len(data)) + 1):
        if data.endswith(_YAML_PREFIX[:prefix_len]):
            max_hold = max(max_hold, prefix_len)
    if max_hold == 0:
        return data, ""
    return data[:-max_hold], data[-max_hold:]


def _parse_yaml_args(body: str) -> Dict[str, Any]:
    """Parse indented ``key: value`` lines into an argument dict."""
    arguments: Dict[str, Any] = {}
    for line in body.splitlines():
        match = _YAML_ARG_LINE.match(line)
        if not match:
            continue
        key = match.group(2)
        value = match.group(3).strip()
        if value.startswith("[Tool call"):
            break
        if value.endswith("]") and value.count("[") == 0:
            value = value[:-1].strip()
        arguments[key] = value
    return arguments


def extract_yaml_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Remove YAML-style ``[Tool call: ...]`` blocks from assistant text.

    Args:
        text: Visible assistant text that may contain inline tool markup.

    Returns:
        Tuple of ``(cleaned_text, tool_dicts)``.
    """
    if _YAML_PREFIX not in text:
        return text, []

    cleaned_parts: List[str] = []
    tools: List[Dict[str, Any]] = []
    pos = 0

    while True:
        match = _YAML_TOOL_START.search(text, pos)
        if match is None:
            cleaned_parts.append(text[pos:])
            break

        if match.start() > pos:
            cleaned_parts.append(text[pos : match.start()])

        name = match.group(1)
        body_start = match.end()
        next_match = _YAML_TOOL_START.search(text, body_start)
        body = text[body_start : next_match.start()] if next_match else text[body_start:]

        args = _parse_yaml_args(body)
        tools.append(normalize_tool_call({"name": name, "arguments": args}))
        pos = next_match.start() if next_match else len(text)

    cleaned = "".join(cleaned_parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, tools


class YamlToolCallProcessor:
    """Incrementally extract YAML-style tool calls from streamed text."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> Tuple[str, List[Dict[str, Any]]]:
        if not chunk:
            return "", []

        self._buffer += chunk
        text_out: List[str] = []
        tools_out: List[Dict[str, Any]] = []

        while True:
            match = _YAML_TOOL_START.search(self._buffer)
            if match is None:
                safe, held = _holdback_yaml_prefix(self._buffer)
                if safe:
                    text_out.append(safe)
                self._buffer = held
                break

            if match.start() > 0:
                text_out.append(self._buffer[: match.start()])
                self._buffer = self._buffer[match.start() :]

            next_match = _YAML_TOOL_START.search(self._buffer, match.end())
            body_start = match.end()
            body = (
                self._buffer[body_start : next_match.start()]
                if next_match
                else self._buffer[body_start:]
            )

            if next_match is None and not body.rstrip().endswith("]"):
                # Incomplete trailing tool — wait for more chunks unless we
                # already have arg lines and the stream ends (handled in flush).
                if not _YAML_ARG_LINE.search(body):
                    break

            name = match.group(1)
            args = _parse_yaml_args(body)
            tools_out.append(normalize_tool_call({"name": name, "arguments": args}))

            if next_match is None:
                self._buffer = ""
                break

            self._buffer = self._buffer[next_match.start() :]

        return "".join(text_out), tools_out

    def flush(self) -> Tuple[str, List[Dict[str, Any]]]:
        if not self._buffer:
            return "", []

        remaining = self._buffer
        self._buffer = ""
        cleaned, tools = extract_yaml_tool_calls(remaining)
        return cleaned, tools
