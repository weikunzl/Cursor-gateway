"""
Parse DeepSeek-native tool-call special tokens embedded in Cursor model output.

DeepSeek family models (V3, V3.1, V3.2, R1, distills, ...) emit tool calls
using their tokenizer's special tokens::

    <｜tool▁calls▁begin｜>
        <｜tool▁call▁begin｜>{type}<｜tool▁sep｜>{name}
        ```json
        {arguments_as_json}
        ```
        <｜tool▁call▁end｜>
        ...
    <｜tool▁calls▁end｜>

The two literal characters that look like an ASCII pipe and underscore are
actually ``｜`` (U+FF5C FULLWIDTH VERTICAL LINE) and ``▁`` (U+2581 LOWER ONE
EIGHTH BLOCK). They are reserved DeepSeek tokens that the model never produces
inside arbitrary text, which is what makes them safe to use as out-of-band
markers.

Cursor's backend passes these tokens through verbatim as plain text in the
unified chat response. Anthropic-compatible clients (Claude Code, Claude
Desktop, etc.) expect structured ``tool_use`` content blocks instead, so we
intercept the markers here and reconstruct the structured calls.

Three body shapes are accepted to cover every DeepSeek variant we've seen in
the wild:

1. **DeepSeek typed-JSON body** (V3 spec)::

       function<｜tool▁sep｜>get_weather
       ```json
       {"city": "Hangzhou"}
       ```

2. **DeepSeek simple body** (V3.1 spec)::

       get_weather<｜tool▁sep｜>{"city": "Brasilia"}

3. **Cursor passthrough key/value body** (seen with DeepSeek-V3.2 routed
   through Cursor)::

       Skill
       <｜tool▁sep｜>skill_name
       devops:my-requirements

The parser sniffs the body to dispatch to the right shape; if none match,
the body is treated as a name-only invocation so we never silently swallow
content.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Build markers from escape sequences so the source code never carries the
# real Unicode characters (which read identically to ASCII look-alikes and
# are easy to corrupt with copy/paste).
_PIPE = "\uff5c"   # ｜  FULLWIDTH VERTICAL LINE
_UBAR = "\u2581"   # ▁  LOWER ONE EIGHTH BLOCK (used by DeepSeek as `_`)

# Marker for [Tool Call: Name({...})] blocks (composer-2.5 / Claude Code format).
_TOOL_CALL_PREFIX = "[Tool Call:"

TOOL_CALLS_BEGIN = f"<{_PIPE}tool{_UBAR}calls{_UBAR}begin{_PIPE}>"
TOOL_CALLS_END = f"<{_PIPE}tool{_UBAR}calls{_UBAR}end{_PIPE}>"
TOOL_CALL_BEGIN = f"<{_PIPE}tool{_UBAR}call{_UBAR}begin{_PIPE}>"
TOOL_CALL_END = f"<{_PIPE}tool{_UBAR}call{_UBAR}end{_PIPE}>"
TOOL_ARG_SEP = f"<{_PIPE}tool{_UBAR}sep{_PIPE}>"

# Sorted longest-first so the streaming holdback prefers the most specific
# marker when chunks land mid-token.
_MARKER_PREFIXES: Tuple[str, ...] = tuple(
    sorted(
        {TOOL_CALLS_BEGIN, TOOL_CALLS_END, TOOL_CALL_BEGIN, TOOL_CALL_END, TOOL_ARG_SEP},
        key=len,
        reverse=True,
    )
)


@dataclass
class ParsedToolCall:
    """A single tool call extracted from passthrough DeepSeek markup."""

    name: str
    arguments: Dict[str, Any]


def _feed_claude_code_tool_calls(
    buffer: str,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Streaming-safe extraction of ``[Tool Call: Name({...})]`` blocks.

    Returns:
        Tuple of ``(remaining_buffer, text_parts, tool_dicts)``. Text parts
        are safe to emit immediately. The remaining buffer may contain a
        partial ``[Tool Call:...`` prefix that will be completed after more
        chunks arrive.
    """
    tools: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    # Only attempt extraction when we see an opening marker.
    if _TOOL_CALL_PREFIX not in buffer:
        # The buffer might still contain DeepSeek markers that need Phase 2
        # processing.  Only hold back a trailing partial [Tool Call: prefix;
        # return the rest to the buffer for the next phase.
        safe, held = _holdback_cc_prefix(buffer)
        # But we must not emit deep-seek markers as text — return them to
        # the buffer so Phase 2 can consume them.
        if held:
            # Partial prefix held back; everything before it goes back.
            # But we don't know where the CC prefix starts — the holdback
            # logic from _holdback_cc_prefix only checks the tail.
            # Actually just keep the buffer intact except for the holdback.
            return held, text_parts, tools
        else:
            # No partial CC prefix — whole buffer is safe from CC perspective.
            # Return it as-is for Phase 2 (DeepSeek) processing.
            return buffer, text_parts, tools

    # Find every occurrence of the marker and process complete tool calls
    # that sit before the last (potentially partial) marker.
    #
    # Strategy: repeatedly find the first occurrence. If the text between
    # it and the end of the buffer contains a complete ``[Tool Call: ...]]``,
    # extract it. Otherwise hold back from that marker onward.
    pos = 0
    out_parts: List[str] = []

    while True:
        marker_idx = buffer.find(_TOOL_CALL_PREFIX, pos)
        if marker_idx == -1:
            # No more markers; rest is safe (with holdback for partial prefix).
            remaining = buffer[pos:]
            safe, held = _holdback_cc_prefix(remaining)
            if safe:
                out_parts.append(safe)
            if out_parts:
                text_parts.append("".join(out_parts))
            return held, text_parts, tools

        # Emit text before the marker.
        if marker_idx > pos:
            prefix = buffer[pos:marker_idx]
            safe, held = _holdback_cc_prefix(prefix)
            if safe:
                out_parts.append(safe)
            if held:
                # Partial prefix — hold back from here.
                candidate = held + buffer[marker_idx:]
                if out_parts:
                    text_parts.append("".join(out_parts))
                return candidate, text_parts, tools

        # Try to extract a complete tool call starting at marker_idx.
        suffix = buffer[marker_idx:]
        cleaned, extracted = _extract_claude_code_tool_calls(suffix)
        if extracted:
            # Got a complete tool call! Emit any preceding text from
            # `cleaned` and advance past it.
            tools.extend(extracted)
            pos = marker_idx + len(suffix) - len(cleaned)
            continue

        # Not a complete tool call yet — hold the suffix for more chunks.
        if out_parts:
            text_parts.append("".join(out_parts))
        return suffix, text_parts, tools


def _holdback_cc_prefix(data: str) -> Tuple[str, str]:
    """Hold back a suffix that may be the start of ``[Tool Call:``.

    Also holds back the entire buffer if it contains a ``[Tool Call:`` marker
    that has not been completed yet (no closing ``)]`` found).
    """
    if not data:
        return "", ""

    marker = _TOOL_CALL_PREFIX

    # If the buffer contains [Tool Call: but no complete tool call is found,
    # hold everything back.
    if marker in data:
        _, extracted = _extract_claude_code_tool_calls(data)
        if not extracted:
            # Incomplete tool call — hold the whole buffer.
            return "", data

    # Also guard against trailing prefixes of the marker.
    for prefix_len in range(min(len(marker), len(data)), 0, -1):
        if data.endswith(marker[:prefix_len]):
            return data[:-prefix_len], data[-prefix_len:]

    return data, ""


def _holdback_suffix(data: str) -> Tuple[str, str]:
    """
    Split ``data`` into a safe-to-emit prefix and a suffix that may still
    complete a marker on the next chunk.

    Args:
        data: Buffered text not yet emitted downstream.

    Returns:
        Tuple ``(safe_prefix, held_suffix)`` where ``held_suffix`` may grow
        into a marker after future chunks arrive.
    """
    if not data:
        return "", ""

    max_hold = 0
    for marker in _MARKER_PREFIXES:
        # Iterate up to the marker length; any partial prefix match means we
        # might complete the marker once more bytes arrive.
        for prefix_len in range(1, min(len(marker), len(data)) + 1):
            if data.endswith(marker[:prefix_len]):
                if prefix_len > max_hold:
                    max_hold = prefix_len

    if max_hold == 0:
        return data, ""
    return data[:-max_hold], data[-max_hold:]


def _try_parse_typed_json(stripped: str) -> Optional[ParsedToolCall]:
    """
    Try the DeepSeek V3 typed body shape.

    Expected body::

        {type}<｜tool▁sep｜>{name}
        ```json
        {arguments_as_json}
        ```

    Args:
        stripped: Tool-call body, already ``.strip()``-ed.

    Returns:
        ``ParsedToolCall`` if the body matches the typed shape, ``None`` if
        it does not so the caller can try other shapes.
    """
    if TOOL_ARG_SEP not in stripped:
        return None

    type_part, _, rest = stripped.partition(TOOL_ARG_SEP)
    if not rest or "\n" not in rest:
        return None

    name_line, _, payload = rest.partition("\n")
    name = name_line.strip()
    if not name:
        return None

    payload = payload.strip()
    if not (payload.startswith("```json") or payload.startswith("```")):
        return None

    # Strip leading fence (``` or ```json) and trailing ```
    if payload.startswith("```json"):
        body = payload[len("```json"):].lstrip("\n")
    else:
        body = payload[len("```"):].lstrip("\n")

    if not body.endswith("```"):
        return None
    body = body[: -len("```")].rstrip()

    try:
        args = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(args, dict):
        # DeepSeek's spec says arguments are always JSON objects; non-objects
        # almost certainly mean we mis-detected the shape.
        return None

    # ``type_part`` (e.g. "function") is recorded only via dispatch; Anthropic
    # tool_use blocks don't carry a separate type field.
    _ = type_part
    return ParsedToolCall(name=name, arguments=args)


def _try_parse_simple_json(stripped: str) -> Optional[ParsedToolCall]:
    """
    Try the DeepSeek V3.1 simplified body shape.

    Expected body::

        {name}<｜tool▁sep｜>{arguments_as_json}

    Args:
        stripped: Tool-call body, already ``.strip()``-ed.

    Returns:
        ``ParsedToolCall`` if the body parses as a name + JSON pair, ``None``
        otherwise.
    """
    if stripped.count(TOOL_ARG_SEP) != 1:
        return None

    name_part, _, value_part = stripped.partition(TOOL_ARG_SEP)
    name = name_part.strip()
    value = value_part.strip()
    if not name or not value:
        return None

    # Only accept this shape if the value parses as a JSON object. This avoids
    # mis-classifying the Cursor key/value shape (which uses the separator as
    # a list delimiter, not a name/value delimiter).
    try:
        args = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(args, dict):
        return None
    return ParsedToolCall(name=name, arguments=args)


def _parse_key_value_body(stripped: str) -> ParsedToolCall:
    """
    Parse the Cursor passthrough key/value body shape.

    Expected body::

        {name}
        <｜tool▁sep｜>{arg_key1}
        {arg_value1}
        <｜tool▁sep｜>{arg_key2}
        {arg_value2}

    Args:
        stripped: Tool-call body, already ``.strip()``-ed.

    Returns:
        ``ParsedToolCall`` with the extracted name and string-valued
        arguments. Unknown / malformed sections are skipped rather than
        raising so partial corruption never tanks the whole response.
    """
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


def _parse_single_tool_call(body: str) -> ParsedToolCall:
    """
    Parse a single ``<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>`` body.

    Dispatches to the best-matching DeepSeek body shape. Order matters: the
    typed-JSON shape is the most specific, then the simple-JSON shape, and
    finally the Cursor key/value shape as a permissive fallback.

    Args:
        body: Raw text between the call begin and end markers.

    Returns:
        ``ParsedToolCall``. An unparseable body returns a call with empty
        name; callers filter those out before emitting tool_use blocks.
    """
    stripped = body.strip()
    if not stripped:
        return ParsedToolCall(name="", arguments={})

    for parser in (_try_parse_typed_json, _try_parse_simple_json):
        parsed = parser(stripped)
        if parsed is not None:
            return parsed

    return _parse_key_value_body(stripped)


def _parse_tool_calls_block(block: str) -> List[ParsedToolCall]:
    """Parse every tool call inside a single ``tool_calls`` envelope."""
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


# Regex to extract [Tool Call: Name({...})] blocks from composable model output.
# Claude Code SDK embeds tool calls in this format; composer-2.5 may return them
# as plain text inside the thinking/content stream rather than as structured
# protobuf fields.
_TOOL_CALL_RE = re.compile(
    r"\[Tool Call:\s+(\w+)\s*\(",  # "[Tool Call: Name("
)


def _extract_claude_code_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract and remove ``[Tool Call: Name({...})]`` blocks from text.

    These blocks appear in composer-2.5 output when the model is simulating
    Claude Code's internal tool-call format. We lift them into structured
    tool-use dicts so downstream clients see proper Anthropic tool_use blocks
    rather than raw markup.

    Returns:
        Tuple of ``(cleaned_text, tool_dicts)``.
    """
    tools: List[Dict[str, Any]] = []
    cleaned_parts: List[str] = []
    pos = 0

    for m in _TOOL_CALL_RE.finditer(text):
        name = m.group(1)
        start = m.start()
        body_start = m.end()

        # Emit everything before this marker
        if start > pos:
            cleaned_parts.append(text[pos:start])

        # Try to find the matching closing bracket: ] preceded by )
        # Format: [Tool Call: Name({...json...})]
        # We need balanced-bracket matching within the JSON payload.
        depth = 0
        in_string = False
        escape_next = False
        end_idx = -1

        for i, ch in enumerate(text[body_start:], start=body_start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            elif ch == ')':
                # Look ahead for ]
                if depth == 0 and i + 1 < len(text) and text[i + 1] == ']':
                    end_idx = i + 2  # skip both )]
                    break

        if end_idx == -1:
            # Unterminated; treat as plain text.
            pos = start
            break

        json_part = text[body_start:end_idx - 2]  # strip trailing )]
        args: Dict[str, Any] = {}
        try:
            parsed = json.loads(json_part)
            if isinstance(parsed, dict):
                args = parsed
        except (json.JSONDecodeError, ValueError):
            pass

        tools.append({"name": name, "arguments": args})
        pos = end_idx

    if pos < len(text):
        cleaned_parts.append(text[pos:])

    return "".join(cleaned_parts).strip(), tools


def extract_redacted_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Remove DeepSeek tool-call markup from ``text`` and return structured calls.

    Args:
        text: Assistant visible text that may carry one or more
            ``<｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜>`` envelopes
            or ``[Tool Call: Name({...})]`` blocks.

    Returns:
        Tuple of ``(cleaned_text, tool_dicts)``. Each tool dict has the
        shape ``{"name": str, "arguments": dict}``. If no markers are found
        the input text is returned unchanged.
    """
    # Phase 1: extract [Tool Call: xxx({...})] blocks (Claude Code / composer-2.5 format)
    text, claude_code_tools = _extract_claude_code_tool_calls(text)

    # Phase 2: extract <｜tool▁calls▁begin｜>... blocks (DeepSeek format)
    if TOOL_CALLS_BEGIN not in text:
        if claude_code_tools:
            return text, claude_code_tools
        return text, []

    cleaned_parts: List[str] = []
    tools: List[Dict[str, Any]] = []
    remainder = text

    while TOOL_CALLS_BEGIN in remainder:
        begin_idx = remainder.find(TOOL_CALLS_BEGIN)
        if begin_idx > 0:
            cleaned_parts.append(remainder[:begin_idx])

        remainder = remainder[begin_idx + len(TOOL_CALLS_BEGIN):]
        # ``rfind`` so a stray inner ``tool_calls_end`` does not truncate the
        # outer block prematurely.
        end_idx = remainder.rfind(TOOL_CALLS_END)
        if end_idx == -1:
            # Unterminated envelope: keep the raw text so callers can see it.
            cleaned_parts.append(TOOL_CALLS_BEGIN + remainder)
            remainder = ""
            break

        block = remainder[:end_idx]
        remainder = remainder[end_idx + len(TOOL_CALLS_END):]

        for parsed in _parse_tool_calls_block(block):
            tools.append({"name": parsed.name, "arguments": parsed.arguments})

    if remainder:
        cleaned_parts.append(remainder)

    cleaned = "".join(cleaned_parts)
    return cleaned, tools





class RedactedToolStreamProcessor:
    """
    Incrementally extract DeepSeek and Claude Code tool calls from streamed
    assistant text.

    The processor emits safe text prefixes immediately and buffers any
    in-progress tool-call envelope or ``[Tool Call: ...`` prefix until the
    closing marker arrives. Partial marker characters at the end of a chunk
    are held back so a marker split across chunk boundaries is still
    recognised once the next chunk arrives.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a streamed text chunk.

        Args:
            chunk: New visible text from the model stream.

        Returns:
            Tuple ``(text_to_emit_now, tool_calls_parsed_now)``. Either side
            may be empty when the chunk only contained partial markers or
            partial tool-call bodies.
        """
        if not chunk:
            return "", []

        self._buffer += chunk
        text_out: List[str] = []
        tools_out: List[Dict[str, Any]] = []

        # Phase 1: handle [Tool Call: Name({...})] blocks (composer-2.5 format).
        # This must happen before DeepSeek marker processing and before the
        # holdback logic, otherwise partial [Tool Call: prefixes get emitted
        # as text prematurely.
        self._buffer, text_parts, cc_tools = _feed_claude_code_tool_calls(
            self._buffer
        )
        if text_parts:
            text_out.extend(text_parts)
        tools_out.extend(cc_tools)

        # Phase 2: handle <｜tool▁calls▁begin｜> blocks (DeepSeek format)
        while TOOL_CALLS_BEGIN in self._buffer:
            begin_idx = self._buffer.find(TOOL_CALLS_BEGIN)
            prefix = self._buffer[:begin_idx]
            if prefix:
                # Run the prefix through CC tool-call extraction first, since
                # it may contain [Tool Call: ...] preceding DeepSeek markers.
                prefix, prefix_parts, prefix_tools = _feed_claude_code_tool_calls(prefix)
                if prefix_parts:
                    text_out.extend(prefix_parts)
                tools_out.extend(prefix_tools)
                if prefix:
                    text_out.append(prefix)

            self._buffer = self._buffer[begin_idx:]
            end_idx = self._buffer.find(TOOL_CALLS_END)
            if end_idx == -1:
                break

            segment = self._buffer[: end_idx + len(TOOL_CALLS_END)]
            self._buffer = self._buffer[end_idx + len(TOOL_CALLS_END):]

            _, parsed = extract_redacted_tool_calls(segment)
            tools_out.extend(parsed)

        # Phase 3: hold back partial markers from the buffer tail.
        #
        # Three cases to handle:
        # 1. Partial [Tool Call: prefix → CC-aware holdback
        # 2. Partial DeepSeek marker (<｜tool...) at tail → DS holdback
        # 3. TOOL_CALLS_BEGIN present but TOOL_CALLS_END absent → hold everything
        if (TOOL_CALLS_BEGIN in self._buffer and
                TOOL_CALLS_END not in self._buffer):
            # Unterminated DeepSeek envelope — hold the entire buffer.
            # Don't emit anything.
            pass
        else:
            safe_cc, held_cc = _holdback_cc_prefix(self._buffer)
            if held_cc:
                if safe_cc:
                    text_out.append(safe_cc)
                self._buffer = held_cc
            else:
                safe, held = _holdback_suffix(self._buffer)
                if safe:
                    text_out.append(safe)
                self._buffer = held

        # Emit text that definitely has no tool markers in it.
        final_text = "".join(text_out)
        return final_text, tools_out

    def flush(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Drain any remaining buffer at end of stream.

        Unterminated tool markup is emitted as plain text rather than
        silently dropped, so the user can still see what the model said.

        Returns:
            Tuple ``(remaining_text, remaining_tool_calls)``.
        """
        if not self._buffer:
            return "", []

        remaining = self._buffer
        self._buffer = ""
        cleaned, tools = extract_redacted_tool_calls(remaining)
        return cleaned, tools
