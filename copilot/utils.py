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

"""
Utility functions for Copilot Gateway.

Contains functions for ID generation and other common utilities.
"""

import uuid


def get_copilot_headers(token: str) -> dict:
    """Builds headers for Copilot API requests."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Editor-Version": "vscode/1.95.0",
        "Editor-Plugin-Version": "copilot-chat/0.22.4",
        "Copilot-Integration-Id": "vscode-chat",
        "User-Agent": "GitHubCopilotChat/0.22.4",
        "openai-intent": "conversation-panel",
    }


def generate_completion_id() -> str:
    """Generates a unique ID for chat completion."""
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_conversation_id() -> str:
    """Generates a random conversation ID."""
    return str(uuid.uuid4())


def generate_tool_call_id() -> str:
    """Generates a unique ID for tool call."""
    return f"call_{uuid.uuid4().hex[:8]}"
