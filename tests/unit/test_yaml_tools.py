"""Tests for YAML-style and ASCII redacted tool-call parsing (session b105efe0)."""

from __future__ import annotations

from cursor.redacted_tools import (
    ASCII_TOOL_CALLS_BEGIN,
    ASCII_TOOL_CALL_BEGIN,
    ASCII_TOOL_CALL_END,
    ASCII_TOOL_CALLS_END,
    ASCII_TOOL_ARG_SEP,
    extract_redacted_tool_calls,
    scrub_leaked_tool_markup,
)
from cursor.yaml_tools import extract_yaml_tool_calls


B105EFE0_BROKEN_TAIL = (
    "\n\n[Tool call: Grep\n"
    "  pattern: _name|set_name|name.*random|randint|secrets|[:8]\n"
    "  path: /home/agilemetrics/workspace/velpos\n"
    "  glob: **/*.{py,js,vue}\n"
    "  head_limit: [Tool call: Grep\n"
    "  pattern: create_session|CreateSession\n"
    "  path: /home/agilemetrics/workspace/velpos\n"
    "  glob: **/*\n"
    "  head_limit: [Tool call: Grep\n"
    "  pattern: newSession|new.*session|session.*name\n"
    "  path: /home/agilemetrics/workspace/velpos/frontend\n"
    "  head_limit: 50\n"
    "]\n\n"
    "</think>\n\n\n\n"
    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>\n"
    "Grep"
)


class TestYamlToolCalls:
    def test_b105efe0_extracts_three_grep_calls(self) -> None:
        cleaned, tools = extract_yaml_tool_calls(B105EFE0_BROKEN_TAIL)
        assert len(tools) == 3
        assert all(t["name"] == "Grep" for t in tools)
        assert tools[0]["arguments"]["pattern"] == "_name|set_name|name.*random|randint|secrets|[:8]"
        assert tools[2]["arguments"]["head_limit"] == "50"
        assert "[Tool call" not in cleaned
        assert "redacted_thinking" not in cleaned

    def test_scrub_removes_ascii_redacted_markers(self) -> None:
        cleaned = scrub_leaked_tool_markup(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>\nGrep"
        )
        assert "redacted_tool" not in cleaned
        assert "Grep" in cleaned


class TestAsciiRedactedToolCalls:
    def test_ascii_envelope_parses_key_value_body(self) -> None:
        block = (
            f"{ASCII_TOOL_CALLS_BEGIN}"
            f"{ASCII_TOOL_CALL_BEGIN}\n"
            "Read\n"
            f"{ASCII_TOOL_ARG_SEP}file_path\n"
            "/tmp/x\n"
            f"{ASCII_TOOL_CALL_END}"
            f"{ASCII_TOOL_CALLS_END}"
        )
        cleaned, tools = extract_redacted_tool_calls(block)
        assert len(tools) == 1
        assert tools[0]["name"] == "Read"
        assert tools[0]["arguments"]["file_path"] == "/tmp/x"
        assert "redacted_tool" not in cleaned

    def test_full_pipeline_cleans_b105efe0_markup(self) -> None:
        cleaned, tools = extract_yaml_tool_calls(B105EFE0_BROKEN_TAIL)
        cleaned, more_tools = extract_redacted_tool_calls(cleaned)
        tools.extend(more_tools)
        cleaned = scrub_leaked_tool_markup(cleaned)
        assert len(tools) >= 3
        assert "redacted_tool" not in cleaned
        assert "[Tool call" not in cleaned
