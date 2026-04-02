"""
Copilot Gateway Configuration.

Centralized storage for all settings, constants, and mappings.
"""

import os
import platform
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.copilot")
# Also try default .env as fallback
load_dotenv()

# ==================================================================================================
# Server Settings
# ==================================================================================================

DEFAULT_SERVER_HOST: str = "0.0.0.0"
SERVER_HOST: str = os.getenv("COPILOT_SERVER_HOST", os.getenv("SERVER_HOST", DEFAULT_SERVER_HOST))

DEFAULT_SERVER_PORT: int = 8002  # Different from Kiro's 8000 and Cursor's 8001
SERVER_PORT: int = int(os.getenv("COPILOT_SERVER_PORT", os.getenv("SERVER_PORT", str(DEFAULT_SERVER_PORT))))

# ==================================================================================================
# Proxy Server Settings
# ==================================================================================================

PROXY_API_KEY: str = os.getenv("PROXY_API_KEY", "my-super-secret-password-123")

# ==================================================================================================
# VPN/Proxy Settings
# ==================================================================================================

VPN_PROXY_URL: str = os.getenv("VPN_PROXY_URL", "")

# ==================================================================================================
# GitHub API Settings
# ==================================================================================================

GITHUB_API_HOST: str = "https://api.github.com"
COPILOT_TOKEN_ENDPOINT: str = "/copilot_internal/v2/token"

COPILOT_API_HOST: str = "https://api.individual.githubcopilot.com"
COPILOT_CHAT_ENDPOINT: str = "/chat/completions"
COPILOT_MODELS_ENDPOINT: str = "/models"

# ==================================================================================================
# GitHub Copilot Credentials
# ==================================================================================================

# Direct GitHub token (personal access token or OAuth token)
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")


def _default_vscode_db_path() -> str:
    """Returns platform-specific default path to VS Code's state.vscdb."""
    system = platform.system()
    if system == "Darwin":
        return str(Path.home() / "Library" / "Application Support" / "Code" / "User" / "globalStorage" / "state.vscdb")
    elif system == "Linux":
        return str(Path.home() / ".config" / "Code" / "User" / "globalStorage" / "state.vscdb")
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return str(Path(appdata) / "Code" / "User" / "globalStorage" / "state.vscdb")
    return ""


COPILOT_VSCODE_DB_PATH: str = os.getenv("COPILOT_VSCODE_DB_FILE", _default_vscode_db_path())

# ==================================================================================================
# Token Settings
# ==================================================================================================

# Copilot tokens are valid for ~30 minutes; refresh when less than 5 minutes remain
TOKEN_REFRESH_THRESHOLD: int = 300

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
    {"modelId": "gpt-4o"},
    {"modelId": "gpt-4o-mini"},
    {"modelId": "claude-sonnet-4"},
    {"modelId": "claude-3.5-sonnet"},
    {"modelId": "o1"},
    {"modelId": "o3-mini"},
]

# ==================================================================================================
# Logging Settings
# ==================================================================================================

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ==================================================================================================
# Application Version
# ==================================================================================================

APP_VERSION: str = "1.0"
APP_TITLE: str = "Copilot Gateway"
APP_DESCRIPTION: str = "Proxy gateway for GitHub Copilot API. OpenAI compatible."
