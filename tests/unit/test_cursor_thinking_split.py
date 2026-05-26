"""Tests for Cursor thinking stream splitter."""

from cursor.thinking_split import CursorThinkingSplitter, split_cursor_thinking_text


class TestCursorThinkingSplitter:
    def test_splits_on_redacted_thinking_tag(self) -> None:
        splitter = CursorThinkingSplitter()
        reasoning, visible = splitter.feed(
            "Plan here.</think>\n\nHello world"
        )
        assert reasoning == "Plan here."
        assert visible == "Hello world"
        assert splitter.flush() == ("", "")

    def test_splits_across_chunks(self) -> None:
        splitter = CursorThinkingSplitter()
        r1, v1 = splitter.feed("Plan ")
        r2, v2 = splitter.feed("more</think>\n\nHi")
        assert r1 == "Plan "
        assert v1 == ""
        assert r2 == "more"
        assert v2 == "Hi"
        assert splitter.flush() == ("", "")

    def test_buffers_partial_close_tag(self) -> None:
        splitter = CursorThinkingSplitter()
        r1, v1 = splitter.feed("text</think>")
        r2, v2 = splitter.feed("\n\nAnswer")
        assert r1 == "text"
        assert v1 == ""
        assert r2 == ""
        assert v2.strip() == "Answer"

    def test_all_thinking_when_no_close_tag(self) -> None:
        splitter = CursorThinkingSplitter()
        reasoning, visible = splitter.feed("only reasoning")
        assert reasoning == "only reasoning"
        assert visible == ""
        assert splitter.flush() == ("", "")

    def test_split_complete_string_helper(self) -> None:
        reasoning, visible = split_cursor_thinking_text(
            "Reasoning.</think>Answer"
        )
        assert "Reasoning" in reasoning
        assert visible.strip() == "Answer"
