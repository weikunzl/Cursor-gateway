"""Tests for Cursor passthrough DeepSeek tool-call parsing.

These tests pin the parser to DeepSeek's native tool-call special tokens
(U+FF5C ``｜`` and U+2581 ``▁``) as observed in real Cursor traffic when the
backend is a DeepSeek family model (V3, V3.1, V3.2, R1, ...). All marker
strings in this file are built from escape sequences so the source code never
contains the lookalike characters by accident.
"""

import json

from cursor.redacted_tools import (
    RedactedToolStreamProcessor,
    TOOL_ARG_SEP,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALLS_BEGIN,
    TOOL_CALLS_END,
    extract_redacted_tool_calls,
)

PIPE = "\uff5c"          # ｜  FULLWIDTH VERTICAL LINE
UBAR = "\u2581"          # ▁  LOWER ONE EIGHTH BLOCK (DeepSeek separator)


# ---------------------------------------------------------------------------
# Marker constants must match the DeepSeek native special tokens exactly.
# ---------------------------------------------------------------------------


class TestMarkerConstants:
    def test_tool_calls_begin_matches_deepseek_spec(self) -> None:
        assert TOOL_CALLS_BEGIN == f"<{PIPE}tool{UBAR}calls{UBAR}begin{PIPE}>"

    def test_tool_calls_end_matches_deepseek_spec(self) -> None:
        assert TOOL_CALLS_END == f"<{PIPE}tool{UBAR}calls{UBAR}end{PIPE}>"

    def test_tool_call_begin_matches_deepseek_spec(self) -> None:
        assert TOOL_CALL_BEGIN == f"<{PIPE}tool{UBAR}call{UBAR}begin{PIPE}>"

    def test_tool_call_end_matches_deepseek_spec(self) -> None:
        assert TOOL_CALL_END == f"<{PIPE}tool{UBAR}call{UBAR}end{PIPE}>"

    def test_tool_arg_sep_matches_deepseek_spec(self) -> None:
        assert TOOL_ARG_SEP == f"<{PIPE}tool{UBAR}sep{PIPE}>"

    def test_no_ascii_pipe_in_markers(self) -> None:
        """Guard against regressing to ASCII ``|`` lookalike markers."""
        for marker in (
            TOOL_CALLS_BEGIN,
            TOOL_CALLS_END,
            TOOL_CALL_BEGIN,
            TOOL_CALL_END,
            TOOL_ARG_SEP,
        ):
            assert "|" not in marker, marker
            assert "_" not in marker, marker
            assert "redacted" not in marker, marker


# ---------------------------------------------------------------------------
# Real-world sample used by ``extract_redacted_tool_calls``.
# ---------------------------------------------------------------------------


KEY_VALUE_BLOCK = (
    "Searching the workspace.\n\n"
    f"{TOOL_CALLS_BEGIN}"
    f"{TOOL_CALL_BEGIN}\n"
    "Grep\n"
    f"{TOOL_ARG_SEP}pattern\n"
    "superpowers\n"
    f"{TOOL_ARG_SEP}glob\n"
    "**/*\n"
    f"{TOOL_CALL_END}"
    f"{TOOL_CALL_BEGIN}\n"
    "SemanticSearch\n"
    f"{TOOL_ARG_SEP}query\n"
    "How to install superpowers\n"
    f"{TOOL_ARG_SEP}num_results\n"
    "15\n"
    f"{TOOL_CALL_END}"
    f"{TOOL_CALLS_END}"
)


USER_REPORTED_BLOCK = (
    "正在使用 ONES 相关技能查询您昨天创建的任务。\n\n"
    f"{TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}\n"
    "Skill\n"
    f"{TOOL_ARG_SEP}skill_name\n"
    "devops:my-requirements\n"
    f"{TOOL_CALL_END}{TOOL_CALLS_END}"
)


class TestExtractRedactedToolCalls:
    def test_user_reported_skill_invocation_is_extracted(self) -> None:
        """Regression test for the exact failure mode the user reported."""
        cleaned, tools = extract_redacted_tool_calls(USER_REPORTED_BLOCK)

        assert PIPE not in cleaned, "Special tokens leaked into visible text"
        assert UBAR not in cleaned, "Special tokens leaked into visible text"
        assert cleaned.strip() == "正在使用 ONES 相关技能查询您昨天创建的任务。"
        assert len(tools) == 1
        assert tools[0]["name"] == "Skill"
        assert tools[0]["arguments"] == {"skill_name": "devops:my-requirements"}

    def test_parses_multiple_key_value_tools(self) -> None:
        cleaned, tools = extract_redacted_tool_calls(KEY_VALUE_BLOCK)

        assert TOOL_CALLS_BEGIN not in cleaned
        assert TOOL_CALLS_END not in cleaned
        assert "Searching the workspace." in cleaned
        assert len(tools) == 2
        assert tools[0]["name"] == "Grep"
        assert tools[0]["arguments"]["pattern"] == "superpowers"
        assert tools[0]["arguments"]["glob"] == "**/*"
        assert tools[1]["name"] == "SemanticSearch"
        assert tools[1]["arguments"]["query"] == "How to install superpowers"
        assert tools[1]["arguments"]["num_results"] == "15"

    def test_returns_original_when_no_markers(self) -> None:
        text = "Hello world, no special tokens here."
        cleaned, tools = extract_redacted_tool_calls(text)
        assert cleaned == text
        assert tools == []

    def test_empty_text(self) -> None:
        cleaned, tools = extract_redacted_tool_calls("")
        assert cleaned == ""
        assert tools == []

    def test_unterminated_block_is_preserved_as_text(self) -> None:
        """A missing closing marker must not silently drop user-visible output."""
        text = f"prefix {TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}\nGrep\n{TOOL_ARG_SEP}p\nq"
        cleaned, tools = extract_redacted_tool_calls(text)
        assert tools == []
        assert "prefix " in cleaned
        assert TOOL_CALLS_BEGIN in cleaned

    def test_deepseek_typed_json_body(self) -> None:
        """DeepSeek V3 spec body: ``{type}<sep>{name}\\n```json\\n{args}\\n``` ``."""
        body = (
            f"{TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}"
            f"function{TOOL_ARG_SEP}get_weather\n"
            "```json\n"
            '{"city": "Hangzhou", "unit": "celsius"}\n'
            "```"
            f"{TOOL_CALL_END}{TOOL_CALLS_END}"
        )
        cleaned, tools = extract_redacted_tool_calls(body)
        assert cleaned == ""
        assert len(tools) == 1
        assert tools[0]["name"] == "get_weather"
        assert tools[0]["arguments"] == {"city": "Hangzhou", "unit": "celsius"}

    def test_deepseek_v31_simple_json_body(self) -> None:
        """DeepSeek V3.1 spec body: ``{name}<sep>{json_args}``."""
        body = (
            f"{TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}"
            f"get_weather{TOOL_ARG_SEP}"
            '{"city": "Brasilia"}'
            f"{TOOL_CALL_END}{TOOL_CALLS_END}"
        )
        cleaned, tools = extract_redacted_tool_calls(body)
        assert cleaned == ""
        assert tools == [{"name": "get_weather", "arguments": {"city": "Brasilia"}}]

    def test_tool_call_with_no_arguments(self) -> None:
        body = (
            f"{TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}\n"
            "ListTasks\n"
            f"{TOOL_CALL_END}{TOOL_CALLS_END}"
        )
        cleaned, tools = extract_redacted_tool_calls(body)
        assert cleaned == ""
        assert tools == [{"name": "ListTasks", "arguments": {}}]

    def test_arguments_containing_marker_lookalikes_are_preserved(self) -> None:
        """Argument values must survive unicode lookalikes inside them."""
        marker_lookalike = "do not parse: " + PIPE + "fake" + PIPE
        body = (
            f"{TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}\n"
            "Echo\n"
            f"{TOOL_ARG_SEP}text\n"
            f"{marker_lookalike}\n"
            f"{TOOL_CALL_END}{TOOL_CALLS_END}"
        )
        cleaned, tools = extract_redacted_tool_calls(body)
        assert cleaned == ""
        assert tools == [{"name": "Echo", "arguments": {"text": marker_lookalike}}]


# ---------------------------------------------------------------------------
# Streaming processor behaviour.
# ---------------------------------------------------------------------------


class TestRedactedToolStreamProcessor:
    def test_streams_complete_block(self) -> None:
        processor = RedactedToolStreamProcessor()
        text, tools = processor.feed(KEY_VALUE_BLOCK)

        assert TOOL_CALLS_BEGIN not in text
        assert len(tools) == 2
        assert processor.flush() == ("", [])

    def test_buffers_incomplete_block(self) -> None:
        processor = RedactedToolStreamProcessor()
        partial = KEY_VALUE_BLOCK[: KEY_VALUE_BLOCK.index(TOOL_CALLS_END)]

        text1, tools1 = processor.feed(partial)
        assert tools1 == []
        assert TOOL_CALLS_BEGIN not in text1

        text2, tools2 = processor.feed(TOOL_CALLS_END)
        assert len(tools2) == 2
        assert processor.flush() == ("", [])

    def test_split_marker_across_chunks(self) -> None:
        processor = RedactedToolStreamProcessor()
        split_at = KEY_VALUE_BLOCK.index(TOOL_CALL_BEGIN)

        text1, tools1 = processor.feed(KEY_VALUE_BLOCK[:split_at])
        text2, tools2 = processor.feed(KEY_VALUE_BLOCK[split_at:])
        text3, tools3 = processor.flush()

        assert tools1 == []
        assert tools3 == []
        assert len(tools2) == 2
        combined = text1 + text2 + text3
        assert TOOL_CALLS_BEGIN not in combined
        assert "Searching the workspace." in combined

    def test_marker_character_split_across_chunks(self) -> None:
        """The opening ``<`` may arrive in a different chunk than the rest."""
        processor = RedactedToolStreamProcessor()
        # Cut exactly between '<' and '｜' of TOOL_CALLS_BEGIN
        begin_idx = USER_REPORTED_BLOCK.index(TOOL_CALLS_BEGIN)
        split_at = begin_idx + 1  # keep the leading '<' in the first chunk

        text1, tools1 = processor.feed(USER_REPORTED_BLOCK[:split_at])
        text2, tools2 = processor.feed(USER_REPORTED_BLOCK[split_at:])

        assert tools1 == []
        combined = text1 + text2
        assert PIPE not in combined
        assert UBAR not in combined
        assert tools2 == [
            {"name": "Skill", "arguments": {"skill_name": "devops:my-requirements"}}
        ]

    def test_chunk_by_chunk_emission_of_user_block(self) -> None:
        """Feeding the user-reported block one character at a time still works."""
        processor = RedactedToolStreamProcessor()
        emitted_text = ""
        tools_seen = []

        for ch in USER_REPORTED_BLOCK:
            text_chunk, tools_chunk = processor.feed(ch)
            emitted_text += text_chunk
            tools_seen.extend(tools_chunk)

        flush_text, flush_tools = processor.flush()
        emitted_text += flush_text
        tools_seen.extend(flush_tools)

        assert PIPE not in emitted_text
        assert UBAR not in emitted_text
        assert emitted_text.strip() == "正在使用 ONES 相关技能查询您昨天创建的任务。"
        assert tools_seen == [
            {"name": "Skill", "arguments": {"skill_name": "devops:my-requirements"}}
        ]

    def test_flush_releases_unterminated_buffer_as_text(self) -> None:
        processor = RedactedToolStreamProcessor()
        partial = f"hello {TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}\nGrep\n"
        text_before, tools_before = processor.feed(partial)
        assert "hello " in text_before
        assert tools_before == []

        flushed_text, flushed_tools = processor.flush()
        # Unterminated block is returned verbatim; nothing silently dropped.
        assert flushed_tools == []
        combined = text_before + flushed_text
        assert TOOL_CALLS_BEGIN in combined
        assert "Grep" in combined

    def test_arguments_with_json_objects_round_trip(self) -> None:
        processor = RedactedToolStreamProcessor()
        args = {"nested": {"a": [1, 2, 3], "b": "x"}}
        body = (
            f"{TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}"
            f"function{TOOL_ARG_SEP}do_thing\n"
            "```json\n"
            f"{json.dumps(args)}\n"
            "```"
            f"{TOOL_CALL_END}{TOOL_CALLS_END}"
        )
        text, tools = processor.feed(body)
        assert text == ""
        assert tools == [{"name": "do_thing", "arguments": args}]
