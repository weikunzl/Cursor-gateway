"""Tests for Cursor redacted tool-call parsing."""

from cursor.redacted_tools import (
    RedactedToolStreamProcessor,
    TOOL_ARG_SEP,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALLS_BEGIN,
    TOOL_CALLS_END,
    extract_redacted_tool_calls,
)

_PIPE = "|"

# Build with ASCII pipe — test file literals may contain Unicode lookalikes
SAMPLE_BLOCK = (
    "Searching the workspace.\n\n"
    f"<{_PIPE}redacted_tool_calls_begin{_PIPE}>"
    f"<{_PIPE}redacted_tool_call_begin{_PIPE}>\n"
    "Grep\n"
    f"<{_PIPE}redacted_tool_sep{_PIPE}>pattern\n"
    "superpowers\n"
    f"<{_PIPE}redacted_tool_sep{_PIPE}>glob\n"
    "**/*\n"
    f"<{_PIPE}redacted_tool_call_end{_PIPE}>"
    f"<{_PIPE}redacted_tool_call_begin{_PIPE}>\n"
    "SemanticSearch\n"
    f"<{_PIPE}redacted_tool_sep{_PIPE}>query\n"
    "How to install superpowers\n"
    f"<{_PIPE}redacted_tool_sep{_PIPE}>num_results\n"
    "15\n"
    f"<{_PIPE}redacted_tool_call_end{_PIPE}>"
    f"<{_PIPE}redacted_tool_calls_end{_PIPE}>"
)


class TestExtractRedactedToolCalls:
    def test_parses_multiple_tools(self) -> None:
        cleaned, tools = extract_redacted_tool_calls(SAMPLE_BLOCK)

        assert TOOL_CALLS_BEGIN not in cleaned
        assert "Searching the workspace." in cleaned
        assert len(tools) == 2
        assert tools[0]["name"] == "Grep"
        assert tools[0]["arguments"]["pattern"] == "superpowers"
        assert tools[0]["arguments"]["glob"] == "**/*"
        assert tools[1]["name"] == "SemanticSearch"
        assert tools[1]["arguments"]["query"] == "How to install superpowers"
        assert tools[1]["arguments"]["num_results"] == "15"

    def test_returns_original_when_no_markers(self) -> None:
        text = "Hello world"
        cleaned, tools = extract_redacted_tool_calls(text)
        assert cleaned == text
        assert tools == []


class TestRedactedToolStreamProcessor:
    def test_streams_complete_block(self) -> None:
        processor = RedactedToolStreamProcessor()
        text, tools = processor.feed(SAMPLE_BLOCK)

        assert TOOL_CALLS_BEGIN not in text
        assert len(tools) == 2
        assert processor.flush() == ("", [])

    def test_buffers_incomplete_block(self) -> None:
        processor = RedactedToolStreamProcessor()
        partial = SAMPLE_BLOCK[: SAMPLE_BLOCK.index(TOOL_CALLS_END)]

        text1, tools1 = processor.feed(partial)
        assert tools1 == []
        assert TOOL_CALLS_BEGIN not in text1

        text2, tools2 = processor.feed(TOOL_CALLS_END)
        assert len(tools2) == 2
        assert processor.flush() == ("", [])

    def test_split_marker_across_chunks(self) -> None:
        processor = RedactedToolStreamProcessor()
        split_at = SAMPLE_BLOCK.index(TOOL_CALL_BEGIN)

        text1, tools1 = processor.feed(SAMPLE_BLOCK[:split_at])
        text2, tools2 = processor.feed(SAMPLE_BLOCK[split_at:])

        assert tools1 == []
        assert len(tools2) == 2
        assert TOOL_CALLS_BEGIN not in text1 + text2
