"""Tests for the ``[Tool Call: Name({json_args})]`` text dialect.

Cursor's ``composer-2.5`` model (routed via cursor-gateway) sometimes narrates
tool invocations in plain Markdown-ish text rather than emitting structured
tool_use blocks or DeepSeek-style special tokens. The format observed in
real traffic looks like::

    [Tool Call: Grep({"pattern": "mcp", "head_limit": "80"})]
    [Tool Call: SemanticSearch({"query": "...", "targetDirectories": []})]

Claude Code can only execute tools delivered as proper Anthropic ``tool_use``
content blocks, so we extract these inline calls into structured form before
they reach the client.

The parser must:
- Recognise the literal ``[Tool Call: `` prefix only.
- Match the closing ``)]`` accounting for balanced parens *and* JSON braces.
- Preserve string literals containing ``(``, ``)``, ``[``, ``]``, ``{``, ``}``.
- Hold back ambiguous suffixes during streaming so a marker split across
  chunks is still recognised.
- Never silently drop visible text — unterminated or malformed calls are
  emitted as-is.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


# Importing the module that does not exist yet is the first thing this test
# file does; this is the explicit "red" step of TDD.
from cursor.bracket_tools import BracketToolCallProcessor, extract_bracket_tool_calls


# ---------------------------------------------------------------------------
# Real-world fixtures (copied verbatim from the user-reported failing run).
# ---------------------------------------------------------------------------


USER_REPORTED_BLOCK = (
    "继续扩大搜索范围，查找 MCP 相关配置与文档。\n\n\n\n"
    '[Tool Call: Grep({"pattern": "mcp", "-i": "true", "head_limit": "80"})]\n'
    '[Tool Call: Glob({"glob_pattern": "**/*mcp*"})]'
)


SEMANTIC_SEARCH_BLOCK = (
    "正在查找工作区或文档中是否有该 MCP 服务的说明；当前会话未连接该 MCP，无法直接列出其工具。\n\n"
    '[Tool Call: SemanticSearch({"query": "internal-mcp MCP server 10.4.27.111 30011",'
    '"targetDirectories":[],"numResults":10})]\n'
    '[Tool Call: WebSearch({"search_term":"Claude Code MCP list tools inspect server capabilities",'
    '"explanation":"Find how to discover MCP server tools after configuration."})]'
)


# ---------------------------------------------------------------------------
# extract_bracket_tool_calls — one-shot batch extraction.
# ---------------------------------------------------------------------------


class TestExtractBracketToolCalls:
    def test_user_reported_block_extracts_two_tools(self) -> None:
        cleaned, tools = extract_bracket_tool_calls(USER_REPORTED_BLOCK)
        assert "[Tool Call:" not in cleaned
        assert cleaned.strip() == "继续扩大搜索范围，查找 MCP 相关配置与文档。"
        assert len(tools) == 2
        assert tools[0] == {
            "name": "Grep",
            "arguments": {"pattern": "mcp", "-i": "true", "head_limit": "80"},
        }
        assert tools[1] == {
            "name": "Glob",
            "arguments": {"glob_pattern": "**/*mcp*"},
        }

    def test_semantic_search_block_with_nested_array_and_quoted_brackets(self) -> None:
        cleaned, tools = extract_bracket_tool_calls(SEMANTIC_SEARCH_BLOCK)
        assert "[Tool Call:" not in cleaned
        assert "正在查找工作区或文档" in cleaned
        assert len(tools) == 2
        assert tools[0]["name"] == "SemanticSearch"
        assert tools[0]["arguments"]["targetDirectories"] == []
        assert tools[0]["arguments"]["numResults"] == 10
        assert tools[1]["name"] == "WebSearch"
        assert "Claude Code" in tools[1]["arguments"]["search_term"]

    def test_no_markers_returns_input_unchanged(self) -> None:
        text = "Just a plain sentence with no tool calls."
        cleaned, tools = extract_bracket_tool_calls(text)
        assert cleaned == text
        assert tools == []

    def test_empty_input(self) -> None:
        cleaned, tools = extract_bracket_tool_calls("")
        assert cleaned == ""
        assert tools == []

    def test_markdown_link_is_not_treated_as_tool_call(self) -> None:
        """``[label](url)`` must not be matched."""
        text = "See the [docs](https://example.com) for details."
        cleaned, tools = extract_bracket_tool_calls(text)
        assert tools == []
        assert cleaned == text

    def test_no_args_object_falls_back_to_empty_dict(self) -> None:
        text = "[Tool Call: PingTool({})]"
        cleaned, tools = extract_bracket_tool_calls(text)
        assert cleaned == ""
        assert tools == [{"name": "PingTool", "arguments": {}}]

    def test_argument_value_contains_parentheses_and_brackets(self) -> None:
        text = '[Tool Call: Echo({"text": "a (b) [c] {d}", "extra": "(yes)"})]'
        cleaned, tools = extract_bracket_tool_calls(text)
        assert cleaned == ""
        assert tools == [
            {
                "name": "Echo",
                "arguments": {"text": "a (b) [c] {d}", "extra": "(yes)"},
            }
        ]

    def test_argument_value_contains_escaped_quotes(self) -> None:
        text = r'[Tool Call: Echo({"text": "She said \"hi\""})]'
        cleaned, tools = extract_bracket_tool_calls(text)
        assert cleaned == ""
        assert tools == [{"name": "Echo", "arguments": {"text": 'She said "hi"'}}]

    def test_deeply_nested_json_object(self) -> None:
        args = {
            "a": {"b": {"c": [1, 2, {"d": "deep"}]}},
            "list": [{"k": "v1"}, {"k": "v2"}],
        }
        text = f'prefix\n[Tool Call: Nested({json.dumps(args)})]\nsuffix'
        cleaned, tools = extract_bracket_tool_calls(text)
        assert tools == [{"name": "Nested", "arguments": args}]
        assert cleaned.replace("\n", " ").strip().startswith("prefix")
        assert cleaned.replace("\n", " ").strip().endswith("suffix")

    def test_unterminated_call_is_kept_as_text(self) -> None:
        text = '[Tool Call: Grep({"pattern": "abc"'
        cleaned, tools = extract_bracket_tool_calls(text)
        assert tools == []
        assert cleaned == text

    def test_malformed_json_skips_and_keeps_text(self) -> None:
        # Missing colon — bad JSON inside the parens.
        text = '[Tool Call: Grep({"pattern" "abc"})]'
        cleaned, tools = extract_bracket_tool_calls(text)
        # We do not invent a tool dict from malformed JSON; we keep the raw text.
        assert tools == []
        assert "[Tool Call:" in cleaned

    def test_multiple_calls_with_text_between(self) -> None:
        text = (
            "Will run two queries.\n"
            '[Tool Call: A({"x":1})]\n'
            "Then a comment.\n"
            '[Tool Call: B({"y":2})]\n'
            "Done."
        )
        cleaned, tools = extract_bracket_tool_calls(text)
        assert [t["name"] for t in tools] == ["A", "B"]
        assert "Then a comment." in cleaned
        assert "Done." in cleaned
        assert "[Tool Call:" not in cleaned

    def test_tool_name_with_underscores_and_digits(self) -> None:
        text = '[Tool Call: tool_v2_run({"x": 1})]'
        cleaned, tools = extract_bracket_tool_calls(text)
        assert tools == [{"name": "tool_v2_run", "arguments": {"x": 1}}]
        assert cleaned == ""


# ---------------------------------------------------------------------------
# BracketToolCallProcessor — streaming behaviour.
# ---------------------------------------------------------------------------


class TestBracketToolCallProcessorStreaming:
    def test_full_block_in_one_chunk(self) -> None:
        p = BracketToolCallProcessor()
        text, tools = p.feed(USER_REPORTED_BLOCK)
        rest_text, rest_tools = p.flush()
        assert (rest_text, rest_tools) == ("", [])
        assert "[Tool Call:" not in text
        assert [t["name"] for t in tools] == ["Grep", "Glob"]

    def test_prefix_split_across_two_chunks(self) -> None:
        """``[Tool C`` then ``all: Foo({})]`` — holdback must reassemble."""
        p = BracketToolCallProcessor()
        text1, tools1 = p.feed("prefix [Tool C")
        text2, tools2 = p.feed('all: Foo({"x":1})]')
        rest_text, rest_tools = p.flush()
        assert tools1 == []
        assert tools2 + rest_tools == [{"name": "Foo", "arguments": {"x": 1}}]
        full_visible = text1 + text2 + rest_text
        assert full_visible.startswith("prefix ")
        assert "[Tool Call:" not in full_visible

    def test_call_split_inside_json_args(self) -> None:
        """Buffer must wait for closing ``)]`` even when JSON spans chunks."""
        p = BracketToolCallProcessor()
        emitted_text = ""
        emitted_tools: List[Dict[str, Any]] = []
        for ch in '[Tool Call: G({"pattern":"abc","glob":"**/*.py"})]':
            t, ts = p.feed(ch)
            emitted_text += t
            emitted_tools.extend(ts)
        ft, fts = p.flush()
        emitted_text += ft
        emitted_tools.extend(fts)
        assert emitted_text == ""
        assert emitted_tools == [
            {"name": "G", "arguments": {"pattern": "abc", "glob": "**/*.py"}}
        ]

    def test_safe_text_emits_immediately(self) -> None:
        """Plain text before any marker must not be buffered indefinitely."""
        p = BracketToolCallProcessor()
        text, tools = p.feed("Hello world! Some long sentence.")
        assert text == "Hello world! Some long sentence."
        assert tools == []
        assert p.flush() == ("", [])

    def test_partial_prefix_at_end_is_held_back(self) -> None:
        """``...words [`` must keep the ``[`` buffered in case marker follows."""
        p = BracketToolCallProcessor()
        text1, tools1 = p.feed("abc [")
        # The trailing '[' is held back.
        assert "[" not in text1
        assert text1 == "abc "
        assert tools1 == []
        # Confirm flushing the held char if no marker arrives.
        ft, fts = p.flush()
        assert ft == "["
        assert fts == []

    def test_partial_prefix_does_not_eat_real_text(self) -> None:
        """``abc [definitely not a marker`` must be fully emitted."""
        p = BracketToolCallProcessor()
        text, tools = p.feed("abc [not a marker yet")
        # We hold back possible-prefix only when it actually matches PREFIX[:n].
        # "[not" cannot extend to "[Tool Call: " so it must flush.
        text += p.flush()[0]
        assert text == "abc [not a marker yet"
        assert tools == []

    def test_unterminated_block_flushes_as_text(self) -> None:
        p = BracketToolCallProcessor()
        text1, tools1 = p.feed('[Tool Call: Grep({"pattern": "abc"')
        ft, fts = p.flush()
        assert tools1 == [] and fts == []
        assert (text1 + ft).startswith("[Tool Call: Grep(")

    def test_chunked_user_reported_block(self) -> None:
        p = BracketToolCallProcessor()
        emitted_text = ""
        emitted_tools: List[Dict[str, Any]] = []
        # 5-char chunks to force splits everywhere.
        for i in range(0, len(USER_REPORTED_BLOCK), 5):
            t, ts = p.feed(USER_REPORTED_BLOCK[i : i + 5])
            emitted_text += t
            emitted_tools.extend(ts)
        ft, fts = p.flush()
        emitted_text += ft
        emitted_tools.extend(fts)
        assert "[Tool Call:" not in emitted_text
        assert emitted_text.strip().startswith("继续扩大搜索范围")
        assert [t["name"] for t in emitted_tools] == ["Grep", "Glob"]


# ---------------------------------------------------------------------------
# Empty-feed contract (matches RedactedToolStreamProcessor's behaviour).
# ---------------------------------------------------------------------------


class TestBracketProcessorEdgeCases:
    def test_empty_feed_is_noop(self) -> None:
        p = BracketToolCallProcessor()
        assert p.feed("") == ("", [])
        assert p.flush() == ("", [])

    def test_multiple_feeds_independent_results(self) -> None:
        p = BracketToolCallProcessor()
        # First chunk: plain text only.
        t1, ts1 = p.feed("just text\n")
        assert t1 == "just text\n"
        assert ts1 == []
        # Second chunk: full marker.
        t2, ts2 = p.feed('[Tool Call: X({"a":1})]')
        ft, fts = p.flush()
        assert t2 + ft == ""
        assert ts2 + fts == [{"name": "X", "arguments": {"a": 1}}]
