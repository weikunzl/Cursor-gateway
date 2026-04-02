"""
Utility functions for Copilot Gateway.

Contains functions for ID generation and other common utilities.
"""

import uuid


def generate_completion_id() -> str:
    """Generates a unique ID for chat completion."""
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_conversation_id() -> str:
    """Generates a random conversation ID."""
    return str(uuid.uuid4())


def generate_tool_call_id() -> str:
    """Generates a unique ID for tool call."""
    return f"call_{uuid.uuid4().hex[:8]}"
