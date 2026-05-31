"""Tests for Claude Code tool argument normalization."""

from __future__ import annotations

import pytest

from cursor.tool_args import normalize_tool_arguments, normalize_tool_call


class TestReadToolNormalization:
    def test_path_maps_to_file_path(self) -> None:
        args = normalize_tool_arguments(
            "Read",
            {"path": "/home/user/file.py"},
        )
        assert args == {"file_path": "/home/user/file.py"}

    def test_file_path_preserved(self) -> None:
        args = normalize_tool_arguments(
            "Read",
            {"file_path": "/home/user/file.py"},
        )
        assert args == {"file_path": "/home/user/file.py"}


class TestGlobToolNormalization:
    def test_glob_pattern_maps_to_pattern(self) -> None:
        args = normalize_tool_arguments(
            "Glob",
            {
                "target_directory": "/home/user/proj",
                "glob_pattern": "**/*.py",
            },
        )
        assert args == {"pattern": "**/*.py", "path": "/home/user/proj"}

    def test_pattern_only(self) -> None:
        args = normalize_tool_arguments(
            "Glob",
            {"glob_pattern": "*.md"},
        )
        assert args == {"pattern": "*.md"}


class TestNormalizeToolCall:
    def test_wraps_name_and_arguments(self) -> None:
        result = normalize_tool_call(
            {"name": "Read", "arguments": {"path": "/tmp/x"}},
        )
        assert result["name"] == "Read"
        assert result["arguments"] == {"file_path": "/tmp/x"}

    @pytest.mark.parametrize("name", ["Grep", "Bash", "Write"])
    def test_unknown_tools_pass_through(self, name: str) -> None:
        raw = {"pattern": "foo", "path": "/tmp"}
        assert normalize_tool_arguments(name, raw) == raw
