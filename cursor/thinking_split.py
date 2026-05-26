"""
Split Cursor thinking stream into reasoning vs user-visible text.

Cursor's composer-2.5 model emits *all* of its output via the protobuf
``thinking`` field, with one of several end-of-reasoning sentinels between
the reasoning trace and the actual answer. We've observed in real traffic:

* HTML-style ``</think>``, ``</thinking>``, ``</reasoning>``, ``</thought>``.
* DeepSeek native tokens such as ``<\uff5cfinal\uff5c>``,
  ``<\uff5cend_of_thinking\uff5c>``, ``<\uff5cend_of_think\uff5c>``,
  ``<\uff5canswer\uff5c>`` (where ``\uff5c`` is the fullwidth vertical line
  ``\uff5c``).

The splitter must recognise *all* of these so the assistant's reply actually
reaches the client. If we miss a sentinel, the entire visible answer ends up
inside the thinking block and the user sees ``Cogitated for Xs`` with no
reply — the bug that triggered this fix.

Defence in depth: any orphan ``<\uff5c<identifier>\uff5c>`` token that still
slips through (e.g. an unknown future sentinel, or a stray emit *after* the
split) is removed from visible text by :class:`DeepSeekTokenScrubber`.
"""

from __future__ import annotations

import re
from typing import List, Tuple

PIPE = "\uff5c"  # ｜  FULLWIDTH VERTICAL LINE

# Longest first so partial matches prefer the full close tag.
CLOSE_TAGS: Tuple[str, ...] = (
    "</thinking>",
    "</reasoning>",
    "</thought>",
    "</think>",
    f"<{PIPE}end_of_thinking{PIPE}>",
    f"<{PIPE}end_of_think{PIPE}>",
    f"<{PIPE}answer{PIPE}>",
    f"<{PIPE}final{PIPE}>",
)


# Defensive scrubber: any DeepSeek-style ``<\uff5cidentifier\uff5c>`` token
# left in visible text. Identifier chars match what DeepSeek actually emits
# (letters, digits, underscore).
_DEEPSEEK_TOKEN_RE = re.compile(rf"<{PIPE}[A-Za-z0-9_]+{PIPE}>")


def _holdback_close_tag_suffix(data: str) -> Tuple[str, str]:
    """Hold back a suffix that may be the start of a partial close tag.

    Args:
        data: Buffered reasoning text not yet flushed.

    Returns:
        ``(safe_prefix, held_suffix)``. ``held_suffix`` is empty when the
        tail of ``data`` cannot conceivably grow into any known close tag.
    """
    if not data:
        return "", ""

    max_hold = 0
    for tag in CLOSE_TAGS:
        for prefix_len in range(1, min(len(tag), len(data)) + 1):
            if data.endswith(tag[:prefix_len]):
                max_hold = max(max_hold, prefix_len)

    if max_hold == 0:
        return data, ""
    return data[:-max_hold], data[-max_hold:]


class CursorThinkingSplitter:
    """Incrementally split a thinking stream into reasoning and visible text.

    Once any recognised sentinel is consumed, subsequent chunks are treated
    as the user-visible answer.
    """

    def __init__(self) -> None:
        self._in_answer: bool = False
        self._carry: str = ""
        # Once we enter answer mode, any further DeepSeek sentinels that
        # the model emits inline are scrubbed so they don't reach the user.
        self._answer_scrubber = DeepSeekTokenScrubber()

    def feed(self, chunk: str) -> Tuple[str, str]:
        """Process one streamed chunk.

        Args:
            chunk: Raw text from the Cursor thinking channel.

        Returns:
            ``(reasoning_text, visible_text)`` for this chunk only. Either
            side may be empty when the chunk is fully reasoning, fully
            visible, or held back as a partial sentinel.
        """
        if not chunk:
            return "", ""

        thinking_parts: List[str] = []
        text_parts: List[str] = []
        data = self._carry + chunk
        self._carry = ""

        while data:
            if self._in_answer:
                text_parts.append(self._answer_scrubber.feed(data))
                data = ""
                continue

            earliest_idx = -1
            matched_tag = ""
            for tag in CLOSE_TAGS:
                idx = data.find(tag)
                if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                    earliest_idx = idx
                    matched_tag = tag

            if earliest_idx != -1:
                before = data[:earliest_idx]
                if before:
                    thinking_parts.append(before)
                # Consume the matched tag plus any immediately-following
                # newlines so the visible body starts cleanly.
                data = data[earliest_idx + len(matched_tag) :].lstrip("\r\n")
                self._in_answer = True
                continue

            safe, held = _holdback_close_tag_suffix(data)
            if safe:
                thinking_parts.append(safe)
            self._carry = held
            break

        return "".join(thinking_parts), "".join(text_parts)

    def flush(self) -> Tuple[str, str]:
        """Drain any remaining buffer at end of stream.

        Returns:
            Whatever was held back is emitted as plain reasoning (when no
            sentinel was ever seen) or as plain visible text (when we're
            already in answer mode). Visible flushes are also scrubbed
            for any held-back partial DeepSeek token.
        """
        if self._in_answer:
            # First drain the carry into the scrubber so it benefits from
            # cross-chunk holdback resolution, then drain the scrubber.
            tail = self._carry
            self._carry = ""
            extra = self._answer_scrubber.feed(tail) if tail else ""
            extra += self._answer_scrubber.flush()
            return "", extra

        if not self._carry:
            return "", ""
        thinking, self._carry = self._carry, ""
        return thinking, ""


def split_cursor_thinking_text(text: str) -> Tuple[str, str]:
    """Split a complete thinking string into reasoning and visible answer.

    Args:
        text: Full thinking field from Cursor.

    Returns:
        ``(reasoning, visible_text)``.
    """
    splitter = CursorThinkingSplitter()
    thinking, visible = splitter.feed(text)
    ft, fv = splitter.flush()
    return thinking + ft, visible + fv


def scrub_deepseek_tokens(text: str) -> str:
    """Remove orphan ``<\uff5c<identifier>\uff5c>`` tokens from ``text``.

    Args:
        text: Visible text that may contain stray DeepSeek sentinels.

    Returns:
        ``text`` with every recognised DeepSeek-style token deleted.
        Standalone fullwidth pipes ``\uff5c`` and ASCII ``|`` are
        preserved.
    """
    if not text:
        return ""
    return _DEEPSEEK_TOKEN_RE.sub("", text)


class DeepSeekTokenScrubber:
    """Stream-friendly version of :func:`scrub_deepseek_tokens`.

    Mirrors the ``feed``/``flush`` interface used elsewhere in the gateway
    (``RedactedToolStreamProcessor``, ``BracketToolCallProcessor``,
    ``CursorThinkingSplitter``) so it can be chained into the visible-text
    pipeline.

    A partial token at the end of a chunk is held back so a token split
    across chunks is still recognised. If the held-back suffix turns out
    not to be a token, it is emitted unchanged on the next ``feed`` /
    ``flush`` call.
    """

    # Max characters we ever need to hold back: ``<\uff5c`` (2) + identifier
    # + ``\uff5c>`` (2). We cap at 64 chars to bound buffer growth in case
    # a malicious / runaway stream contains ``<\uff5c<huge>...`` without
    # the closing ``\uff5c>``.
    _MAX_HOLD = 64

    def __init__(self) -> None:
        self._carry: str = ""

    def feed(self, chunk: str) -> str:
        """Process one streamed visible-text chunk.

        Args:
            chunk: New visible text from upstream.

        Returns:
            Safe-to-emit text. Empty when the whole chunk was held back as
            a potential token prefix.
        """
        if not chunk:
            return ""
        data = self._carry + chunk
        self._carry = ""

        cleaned = _DEEPSEEK_TOKEN_RE.sub("", data)

        # If the tail looks like the start of another token, hold it back.
        idx = cleaned.rfind("<" + PIPE)
        if idx != -1:
            tail = cleaned[idx:]
            # If we already see the closing pipe+">", it's a complete
            # token that just didn't match (unknown identifier shape) —
            # emit verbatim rather than buffering forever.
            if PIPE + ">" not in tail and len(tail) <= self._MAX_HOLD:
                self._carry = tail
                cleaned = cleaned[:idx]
        else:
            # Could the very last character(s) be the start of "<｜"?
            if cleaned.endswith("<"):
                self._carry = "<"
                cleaned = cleaned[:-1]

        return cleaned

    def flush(self) -> str:
        """Emit any held-back partial token verbatim at end of stream."""
        held, self._carry = self._carry, ""
        return held
