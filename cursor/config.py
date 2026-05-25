"""
Cursor Gateway Configuration.

Centralized storage for all settings, constants, and mappings.
"""

import os
import platform
import sys
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.cursor")
# Also try default .env as fallback
load_dotenv()

# ==================================================================================================
# Server Settings
# ==================================================================================================

DEFAULT_SERVER_HOST: str = "0.0.0.0"
SERVER_HOST: str = os.getenv("CURSOR_SERVER_HOST", os.getenv("SERVER_HOST", DEFAULT_SERVER_HOST))

DEFAULT_SERVER_PORT: int = 8001  # Different from Kiro's 8000
SERVER_PORT: int = int(os.getenv("CURSOR_SERVER_PORT", os.getenv("SERVER_PORT", str(DEFAULT_SERVER_PORT))))

# ==================================================================================================
# Proxy Server Settings
# ==================================================================================================

PROXY_API_KEY: str = os.getenv("PROXY_API_KEY", "my-super-secret-password-123")

# ==================================================================================================
# VPN/Proxy Settings
# ==================================================================================================

VPN_PROXY_URL: str = os.getenv("VPN_PROXY_URL", "")

# ==================================================================================================
# Cursor API Settings
# ==================================================================================================

CURSOR_API_HOST: str = os.getenv("CURSOR_API_HOST", "https://api2.cursor.sh")

# ConnectRPC endpoints
CURSOR_CHAT_RPC: str = "/aiserver.v1.ChatService/StreamUnifiedChatWithTools"
CURSOR_MODELS_RPC: str = "/aiserver.v1.AiService/AvailableModels"

# Client identification
CURSOR_CLIENT_VERSION: str = os.getenv("CURSOR_CLIENT_VERSION", "3.2.21")
CURSOR_CLIENT_TYPE: str = "ide"

# Ghost mode (privacy mode) - sends x-ghost-mode header
_GHOST_MODE_RAW: str = os.getenv("CURSOR_GHOST_MODE", "true").lower()
CURSOR_GHOST_MODE: bool = _GHOST_MODE_RAW in ("true", "1", "yes")

# ==================================================================================================
# Cursor Credentials
# ==================================================================================================

# Direct token override (alternative to SQLite)
CURSOR_ACCESS_TOKEN: str = os.getenv("CURSOR_ACCESS_TOKEN", "")

# Machine ID override (alternative to SQLite)
CURSOR_MACHINE_ID: str = os.getenv("CURSOR_MACHINE_ID", "")

# Path to Cursor's state.vscdb SQLite database
def _default_cursor_db_path() -> str:
    """Returns platform-specific default path to Cursor's state.vscdb."""
    system = platform.system()
    if system == "Darwin":
        return str(Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "state.vscdb")
    elif system == "Linux":
        return str(Path.home() / ".config" / "Cursor" / "User" / "globalStorage" / "state.vscdb")
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return str(Path(appdata) / "Cursor" / "User" / "globalStorage" / "state.vscdb")
    return ""

CURSOR_DB_PATH: str = os.getenv("CURSOR_DB_FILE", _default_cursor_db_path())

# ==================================================================================================
# Token Settings
# ==================================================================================================

TOKEN_REFRESH_THRESHOLD: int = 600

# ==================================================================================================
# Retry Configuration
# ==================================================================================================

MAX_RETRIES: int = 3
BASE_RETRY_DELAY: float = 1.0

# ==================================================================================================
# Streaming Settings
# ==================================================================================================

FIRST_TOKEN_TIMEOUT: float = float(os.getenv("FIRST_TOKEN_TIMEOUT", "15"))
STREAMING_READ_TIMEOUT: float = float(os.getenv("STREAMING_READ_TIMEOUT", "300"))
FIRST_TOKEN_MAX_RETRIES: int = int(os.getenv("FIRST_TOKEN_MAX_RETRIES", "3"))

# ==================================================================================================
# Model Settings
# ==================================================================================================

MODEL_CACHE_TTL: int = 3600
DEFAULT_MAX_INPUT_TOKENS: int = 200000

# Model aliases
MODEL_ALIASES: Dict[str, str] = {}

# Models hidden from /v1/models list
HIDDEN_FROM_LIST: List[str] = []

# Fallback models when API is unreachable
FALLBACK_MODELS: List[Dict[str, str]] = [
    {"modelId": "claude-4-sonnet"},
    {"modelId": "claude-3.5-sonnet"},
    {"modelId": "claude-4-opus"},
    {"modelId": "gpt-4o"},
    {"modelId": "gpt-4o-mini"},
    {"modelId": "cursor-small"},
    {"modelId": "gpt-5.3-codex"},
    {"modelId": "claude-4.6-sonnet-medium-thinking"},
    {"modelId": "gpt-5.5-medium"},
    {"modelId": "claude-opus-4-7-thinking-xhign"},
]

# ==================================================================================================
# Logging Settings
# ==================================================================================================

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ==================================================================================================
# Debug Settings
# ==================================================================================================

_DEBUG_MODE_RAW: str = os.getenv("DEBUG_MODE", "").lower()
if _DEBUG_MODE_RAW in ("off", "errors", "all"):
    DEBUG_MODE: str = _DEBUG_MODE_RAW
else:
    DEBUG_MODE: str = "off"

DEBUG_DIR: str = os.getenv("DEBUG_DIR", "debug_logs")

# ==================================================================================================
# Application Version
# ==================================================================================================

APP_VERSION: str = "1.0"
APP_TITLE: str = "Cursor Gateway"
APP_DESCRIPTION: str = "Proxy gateway for Cursor API. OpenAI and Anthropic compatible."
