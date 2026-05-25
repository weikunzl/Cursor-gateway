"""
Utility functions for Cursor Gateway.

Contains functions for header building, ID generation,
and other common utilities.
"""

import platform
import uuid
from typing import TYPE_CHECKING

from loguru import logger

from cursor.checksum import compute_checksum
from cursor.config import (
    CURSOR_CLIENT_VERSION,
    CURSOR_CLIENT_TYPE,
    CURSOR_GHOST_MODE,
)

if TYPE_CHECKING:
    from cursor.auth import CursorAuthManager


def _platform_os() -> str:
    """Returns platform OS string for Cursor headers."""
    system = platform.system()
    if system == "Darwin":
        return "darwin"
    elif system == "Linux":
        return "linux"
    elif system == "Windows":
        return "win32"
    return system.lower()


def _platform_arch() -> str:
    """Returns platform architecture string for Cursor headers."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    elif machine in ("x86_64", "amd64"):
        return "x86_64"
    return machine


def _platform_os_version() -> str:
    """Returns platform OS version string for Cursor headers."""
    if platform.system() == "Darwin":
        return platform.mac_ver()[0] or platform.release()
    return platform.release()


def get_cursor_headers(auth_manager: "CursorAuthManager") -> dict:
    """
    Builds headers for Cursor ConnectRPC API requests.

    Args:
        auth_manager: Authentication manager

    Returns:
        Dictionary with all required headers
    """
    token = auth_manager.get_access_token()
    request_id = str(uuid.uuid4())

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/connect+proto",
        "Connect-Protocol-Version": "1",
        "User-Agent": "connect-es/1.6.1",
        "x-cursor-client-version": CURSOR_CLIENT_VERSION,
        "x-cursor-client-type": CURSOR_CLIENT_TYPE,
        "x-cursor-client-os": _platform_os(),
        "x-cursor-client-arch": _platform_arch(),
        "x-cursor-client-os-version": _platform_os_version(),
        "x-cursor-client-device-type": "desktop",
        "x-cursor-config-version": str(uuid.uuid4()),
        "x-cursor-timezone": "Asia/Shanghai",
        "x-amzn-trace-id": f"Root={request_id}",
        "x-new-onboarding-completed": "true",
        "x-ghost-mode": "true" if CURSOR_GHOST_MODE else "false",
        "x-request-id": request_id,
        "x-session-id": auth_manager.session_id,
        "x-client-key": auth_manager.client_key,
    }

    # Add checksum if machine_id is available
    if auth_manager.machine_id:
        headers["x-cursor-checksum"] = compute_checksum(auth_manager.machine_id)

    return headers


def generate_completion_id() -> str:
    """Generates a unique ID for chat completion."""
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_conversation_id() -> str:
    """Generates a random conversation ID."""
    return str(uuid.uuid4())


def generate_tool_call_id() -> str:
    """Generates a unique ID for tool call."""
    return f"call_{uuid.uuid4().hex[:8]}"


def generate_message_id() -> str:
    """Generates a unique message ID for Cursor protocol."""
    return str(uuid.uuid4())
