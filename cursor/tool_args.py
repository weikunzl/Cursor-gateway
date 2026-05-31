"""
Normalize Cursor/composer tool arguments to Claude Code SDK schemas.

composer-2.5 often emits parameter names that match Cursor's internal tools
(``path``, ``glob_pattern``, ``target_directory``) rather than Claude Code's
(``file_path``, ``pattern``, ``path``). Without this mapping, Velpos/Claude
Code reject tool calls and the agent loop collapses into a short, useless reply.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, MutableMapping


def _coerce_arguments(arguments: Any) -> Dict[str, Any]:
    """Return tool arguments as a dict, parsing JSON strings when needed."""
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def normalize_tool_arguments(name: str, arguments: Any) -> Dict[str, Any]:
    """
    Map composer/Cursor tool argument names to Claude Code expectations.

    Args:
        name: Tool name as emitted by the model (e.g. ``Read``, ``Glob``).
        arguments: Raw arguments dict or JSON string from the stream.

    Returns:
        Normalized argument dict safe to forward to Claude Code clients.
    """
    args: MutableMapping[str, Any] = _coerce_arguments(arguments)
    if not args:
        return dict(args)

    if name == "Read":
        if "file_path" not in args and "path" in args:
            args["file_path"] = args.pop("path")

    elif name == "Glob":
        if "pattern" not in args:
            if "glob_pattern" in args:
                args["pattern"] = args.pop("glob_pattern")
            elif "glob" in args:
                args["pattern"] = args.pop("glob")
        if "target_directory" in args and "path" not in args:
            args["path"] = args.pop("target_directory")
        args.pop("glob_pattern", None)
        args.pop("target_directory", None)

    elif name == "Grep":
        # composer sometimes sends ``glob`` where Claude Code expects ``glob`` — ok.
        # Map ``output_mode`` only when it would be rejected (keep passthrough otherwise).
        pass

    return dict(args)


def normalize_tool_call(tool: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize a tool call dict ``{name, arguments}`` in place logically.

    Args:
        tool: Tool call with ``name`` and ``arguments`` keys.

    Returns:
        New dict with normalized ``arguments``.
    """
    name = str(tool.get("name", ""))
    arguments = tool.get("arguments", tool.get("input", {}))
    return {
        "name": name,
        "arguments": normalize_tool_arguments(name, arguments),
    }
