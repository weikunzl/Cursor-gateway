"""Tests for Cursor thinking stream splitter."""

from cursor.thinking_split import (
    CursorThinkingSplitter,
    split_cursor_thinking_text,
    scrub_deepseek_tokens,
    DeepSeekTokenScrubber,
)


PIPE = "\uff5c"  # ｜ U+FF5C FULLWIDTH VERTICAL LINE
FINAL = f"<{PIPE}final{PIPE}>"
END_OF_THINKING = f"<{PIPE}end_of_thinking{PIPE}>"
END_OF_THINK = f"<{PIPE}end_of_think{PIPE}>"
ANSWER = f"<{PIPE}answer{PIPE}>"


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


class TestDeepSeekFinalSentinel:
    """composer-2.5 sometimes uses DeepSeek native tokens such as
    ``<\uff5cfinal\uff5c>`` to separate reasoning from the visible answer,
    instead of the HTML-style ``</think>``. The splitter must recognise
    these as answer-mode triggers so the user actually sees the reply.
    """

    def test_final_sentinel_alone_switches_to_visible(self) -> None:
        splitter = CursorThinkingSplitter()
        reasoning, visible = splitter.feed(
            f"Thinking about it.{FINAL}\n\nHere's the answer."
        )
        assert reasoning == "Thinking about it."
        assert visible.strip() == "Here's the answer."

    def test_close_think_then_final_consumes_both(self) -> None:
        splitter = CursorThinkingSplitter()
        reasoning, visible = splitter.feed(
            f"reason</think>{FINAL}\n\n## 原因\nstuff"
        )
        assert reasoning == "reason"
        assert "<" not in visible
        assert PIPE not in visible
        assert "原因" in visible

    def test_final_sentinel_split_across_chunks(self) -> None:
        splitter = CursorThinkingSplitter()
        # Send "...<｜fi" then "nal｜>answer"
        r1, v1 = splitter.feed("plan<")
        r2, v2 = splitter.feed(f"{PIPE}fi")
        r3, v3 = splitter.feed(f"nal{PIPE}>answer")
        assert v1 == "" and v2 == ""
        assert v3 == "answer"
        assert (r1 + r2 + r3) == "plan"

    def test_end_of_thinking_token_also_recognised(self) -> None:
        splitter = CursorThinkingSplitter()
        reasoning, visible = splitter.feed(
            f"reasoning text{END_OF_THINKING}visible body"
        )
        assert reasoning == "reasoning text"
        assert visible == "visible body"

    def test_no_sentinel_keeps_full_text_as_reasoning(self) -> None:
        """Behaviour pin: if no recognised sentinel, no visible output."""
        splitter = CursorThinkingSplitter()
        reasoning, visible = splitter.feed("just thinking, no marker")
        ft, fv = splitter.flush()
        assert reasoning + ft == "just thinking, no marker"
        assert visible + fv == ""


class TestDeepSeekTokenScrubber:
    """Belt-and-braces: any orphan ``<\uff5c...\uff5c>`` token that escapes the
    splitter must be removed from visible text before reaching the client.
    """

    def test_strips_inline_token_from_text(self) -> None:
        cleaned = scrub_deepseek_tokens(f"hello {FINAL} world")
        assert cleaned == "hello  world"

    def test_strips_multiple_distinct_tokens(self) -> None:
        cleaned = scrub_deepseek_tokens(f"{ANSWER}A{END_OF_THINK}B{FINAL}C")
        assert cleaned == "ABC"

    def test_preserves_ascii_pipes(self) -> None:
        cleaned = scrub_deepseek_tokens("a | b | c")
        assert cleaned == "a | b | c"

    def test_preserves_non_token_fullwidth_pipes(self) -> None:
        """Standalone ｜ outside a ``<｜id｜>`` shape must stay."""
        cleaned = scrub_deepseek_tokens("分隔符 ｜ 数据")
        assert cleaned == "分隔符 ｜ 数据"

    def test_streaming_holds_back_partial_token(self) -> None:
        scrubber = DeepSeekTokenScrubber()
        out1 = scrubber.feed(f"answer text <")
        out2 = scrubber.feed(f"{PIPE}fi")
        out3 = scrubber.feed(f"nal{PIPE}> more")
        flushed = scrubber.flush()
        assert (out1 + out2 + out3 + flushed) == "answer text  more"

    def test_streaming_partial_at_end_is_flushed_unchanged(self) -> None:
        scrubber = DeepSeekTokenScrubber()
        out = scrubber.feed("safe text < not a token")
        flushed = scrubber.flush()
        assert out + flushed == "safe text < not a token"

    def test_streaming_dangling_partial_is_emitted_on_flush(self) -> None:
        scrubber = DeepSeekTokenScrubber()
        # Stream ends in the middle of a potential token.
        out = scrubber.feed(f"text <{PIPE}par")
        flushed = scrubber.flush()
        assert out + flushed == f"text <{PIPE}par"

    def test_empty_input_noop(self) -> None:
        assert scrub_deepseek_tokens("") == ""
        scrubber = DeepSeekTokenScrubber()
        assert scrubber.feed("") == ""
        assert scrubber.flush() == ""

    def test_token_identifier_allows_underscore_and_digits(self) -> None:
        weird = f"<{PIPE}end_of_thinking_2{PIPE}>"
        cleaned = scrub_deepseek_tokens(f"a{weird}b")
        assert cleaned == "ab"

    def test_does_not_eat_html_tags(self) -> None:
        cleaned = scrub_deepseek_tokens("<think>x</think> y <div>z</div>")
        assert cleaned == "<think>x</think> y <div>z</div>"
