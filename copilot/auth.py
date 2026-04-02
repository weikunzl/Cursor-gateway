"""
Authentication manager for GitHub Copilot API.

Two-step flow:
1. Get GitHub token (env var or VS Code SQLite)
2. Exchange for short-lived Copilot token via GitHub API
"""

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from copilot.config import (
    GITHUB_TOKEN,
    COPILOT_VSCODE_DB_PATH,
    COPILOT_TOKEN_ENDPOINT,
    COPILOT_API_HOST,
    TOKEN_REFRESH_THRESHOLD,
)


class CopilotAuthManager:
    """
    Manages GitHub Copilot authentication tokens.

    Token sources (priority order):
    1. Environment variable (GITHUB_TOKEN)
    2. VS Code SQLite database (state.vscdb)
    """

    def __init__(
        self,
        github_token: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self._github_token = github_token or GITHUB_TOKEN or None
        self._db_path = db_path or COPILOT_VSCODE_DB_PATH or None

        # Load from SQLite if token not provided via env
        if not self._github_token and self._db_path:
            self._load_from_sqlite(self._db_path)

        # Copilot token state
        self._copilot_token: Optional[str] = None
        self._copilot_token_expires_at: float = 0
        self._copilot_api_host: str = COPILOT_API_HOST
        self._lock = asyncio.Lock()

        if self._github_token:
            preview = self._github_token[:12] + "..."
            logger.info(f"Copilot auth initialized: github_token={preview}")
        else:
            logger.warning("Copilot auth: no GitHub token available")

    def _load_from_sqlite(self, db_path: str) -> None:
        """Load GitHub token from VS Code's state.vscdb."""
        try:
            path = Path(db_path).expanduser()
            if not path.exists():
                logger.warning(f"VS Code database not found: {db_path}")
                return

            uri = f"file:{path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            cursor = conn.cursor()

            # Try known keys for GitHub/Copilot token
            for key in [
                "github.copilot.chat.token",
                "github.copilot.token",
                "github-enterprise.copilot.token",
            ]:
                cursor.execute(
                    "SELECT value FROM ItemTable WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                if row and row[0]:
                    raw = row[0]
                    # Token may be JSON string with quotes
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = raw[1:-1]
                    self._github_token = raw
                    logger.info(f"Loaded GitHub token from SQLite (key={key})")
                    break

            conn.close()
        except sqlite3.OperationalError as e:
            logger.error(f"SQLite error reading VS Code DB: {e}")
        except Exception as e:
            logger.error(f"Error loading GitHub token: {e}")

    async def get_copilot_token(self) -> str:
        """
        Returns a valid Copilot API token.
        Refreshes automatically if expired or about to expire.

        Raises:
            ValueError: If no GitHub token is available
        """
        async with self._lock:
            now = time.time()
            if (
                self._copilot_token
                and self._copilot_token_expires_at - now > TOKEN_REFRESH_THRESHOLD
            ):
                return self._copilot_token

            # Need to refresh
            await self._exchange_token()

            if not self._copilot_token:
                raise ValueError(
                    "Failed to obtain Copilot token. "
                    "Check your GitHub token or VS Code login."
                )
            return self._copilot_token

    async def _exchange_token(self) -> None:
        """Exchange GitHub token for Copilot token."""
        if not self._github_token:
            # Try reloading from SQLite
            if self._db_path:
                self._load_from_sqlite(self._db_path)
            if not self._github_token:
                raise ValueError(
                    "No GitHub token available. "
                    "Set GITHUB_TOKEN or log in to VS Code with Copilot."
                )

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    COPILOT_TOKEN_ENDPOINT,
                    headers={
                        "Authorization": f"token {self._github_token}",
                        "Accept": "application/json",
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    self._copilot_token = data.get("token")
                    self._copilot_token_expires_at = data.get("expires_at", 0)

                    # Update API host if provided
                    endpoints = data.get("endpoints", {})
                    if endpoints.get("api"):
                        self._copilot_api_host = endpoints["api"]

                    logger.info(
                        f"Copilot token obtained, expires_at={self._copilot_token_expires_at}, "
                        f"api_host={self._copilot_api_host}"
                    )
                elif response.status_code == 401:
                    logger.error("GitHub token is invalid or expired (401)")
                    # Try reloading from SQLite
                    if self._db_path:
                        self._load_from_sqlite(self._db_path)
                else:
                    logger.error(
                        f"Failed to get Copilot token: HTTP {response.status_code} "
                        f"{response.text[:200]}"
                    )
        except Exception as e:
            logger.error(f"Error exchanging token: {e}")

    async def force_refresh(self) -> None:
        """Force token refresh (e.g., after 401 from upstream)."""
        async with self._lock:
            self._copilot_token = None
            self._copilot_token_expires_at = 0
            await self._exchange_token()

    @property
    def api_host(self) -> str:
        """Copilot API base URL (from token response)."""
        return self._copilot_api_host
