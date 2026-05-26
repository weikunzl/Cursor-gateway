"""
Split Cursor thinking stream into reasoning vs user-visible text.

Cursor's composer models often emit the full assistant reply inside the protobuf
thinking field, delimited by tags such as </think>, instead of a
separate content field. Claude Code expects reasoning in a thinking block and the
answer in a text block.
"""

from typing import List, Tuple

# Longest first so partial matches prefer the full close tag
CLOSE_TAGS: Tuple[str, ...] = (
    "</think>",
    "</thinking>",
    "</reasoning>",
    "</thought>",
)


def _holdback_close_tag_suffix(data: str) -> Tuple[str, str]:
    """Hold back a suffix that may be the start of a partial close tag."""
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
    """
    Incrementally split thinking deltas into (reasoning, visible_text) pairs.

    After the first recognized close tag, subsequent chunks are treated as
    visible answer text until the stream ends.
    """

    def __init__(self) -> None:
        self._in_answer: bool = False
        self._carry: str = ""
        self._max_tag_len: int = max(len(tag) for tag in CLOSE_TAGS)

    def feed(self, chunk: str) -> Tuple[str, str]:
        """
        Process a thinking delta chunk.

        Args:
            chunk: Raw thinking text from Cursor

        Returns:
            Tuple of (reasoning_text, visible_text) for this chunk only
        """
        if not chunk:
            return "", ""

        thinking_parts: List[str] = []
        text_parts: List[str] = []
        data = self._carry + chunk
        self._carry = ""

        while data:
            if self._in_answer:
                text_parts.append(data)
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
        """Flush buffered suffix at end of stream."""
        if not self._carry:
            return "", ""
        if self._in_answer:
            text, self._carry = self._carry, ""
            return "", text
        thinking, self._carry = self._carry, ""
        return thinking, ""


def split_cursor_thinking_text(text: str) -> Tuple[str, str]:
    """
    Split a complete thinking string into reasoning and visible answer.

    Args:
        text: Full thinking field from Cursor

    Returns:
        (reasoning, visible_text)
    """
    splitter = CursorThinkingSplitter()
    thinking, visible = splitter.feed(text)
    ft, fv = splitter.flush()
    return thinking + ft, visible + fv
