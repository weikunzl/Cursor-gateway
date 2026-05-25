"""
Authentication manager for Cursor API.

Manages access tokens loaded from Cursor's local SQLite database
or environment variables.
"""

import hashlib
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from cursor.config import (
    CURSOR_ACCESS_TOKEN,
    CURSOR_MACHINE_ID,
    CURSOR_DB_PATH,
    CURSOR_API_HOST,
)

# UUID namespace for session ID derivation
_UUID_NAMESPACE = uuid.NAMESPACE_DNS


class CursorAuthManager:
    """
    Manages Cursor authentication tokens.

    Token sources (priority order):
    1. Environment variables (CURSOR_ACCESS_TOKEN, CURSOR_MACHINE_ID)
    2. SQLite database (state.vscdb)

    Token format in SQLite: "user_xxxxx::JWT"
    The part after "::" is the Bearer token.
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        machine_id: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self._access_token = access_token or CURSOR_ACCESS_TOKEN or None
        self._refresh_token: Optional[str] = None
        self._machine_id = machine_id or CURSOR_MACHINE_ID or None
        self._email: Optional[str] = None
        self._db_path = db_path or CURSOR_DB_PATH or None

        # Load from SQLite if token not provided via env
        if not self._access_token and self._db_path:
            self._load_from_sqlite(self._db_path)

        # Derive session_id and client_key from token
        self._session_id: Optional[str] = None
        self._client_key: Optional[str] = None
        if self._access_token:
            self._derive_identifiers()

        # Log status
        if self._access_token:
            token_preview = self._access_token[:20] + "..." if len(self._access_token) > 20 else self._access_token
            logger.info(f"Cursor auth initialized: token={token_preview}, machine_id={'set' if self._machine_id else 'not set'}")
        else:
            logger.warning("Cursor auth: no access token available")

    def _load_from_sqlite(self, db_path: str) -> None:
        """Load credentials from Cursor's state.vscdb SQLite database."""
        try:
            path = Path(db_path).expanduser()
            if not path.exists():
                logger.warning(f"Cursor database not found: {db_path}")
                return

            # Open read-only to avoid locking with running Cursor IDE
            uri = f"file:{path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            cursor = conn.cursor()

            # Read access token
            cursor.execute("SELECT value FROM ItemTable WHERE key = ?", ("cursorAuth/accessToken",))
            row = cursor.fetchone()
            if row and row[0]:
                raw_token = row[0]
                # Token format: "user_xxxxx::JWT" - take the JWT part
                if "::" in raw_token:
                    self._access_token = raw_token.split("::", 1)[1]
                else:
                    self._access_token = raw_token
                logger.debug("Loaded access token from SQLite")

            # Read refresh token
            cursor.execute("SELECT value FROM ItemTable WHERE key = ?", ("cursorAuth/refreshToken",))
            row = cursor.fetchone()
            if row and row[0]:
                self._refresh_token = row[0]
                logger.debug("Loaded refresh token from SQLite")

            # Read machine ID
            cursor.execute("SELECT value FROM ItemTable WHERE key = ?", ("storage.serviceMachineId",))
            row = cursor.fetchone()
            if row and row[0]:
                self._machine_id = row[0]
                logger.debug(f"Loaded machine ID from SQLite: {self._machine_id[:8]}...")

            # Read email (informational)
            cursor.execute("SELECT value FROM ItemTable WHERE key = ?", ("cursorAuth/cachedEmail",))
            row = cursor.fetchone()
            if row and row[0]:
                self._email = row[0]
                logger.info(f"Cursor account: {self._email}")

            conn.close()
            logger.info(f"Credentials loaded from Cursor database: {db_path}")

        except sqlite3.OperationalError as e:
            if "unable to open database" in str(e).lower():
                logger.error(f"Cannot open Cursor database (is Cursor installed?): {e}")
            elif "no such table" in str(e).lower():
                logger.error(f"Invalid Cursor database format: {e}")
            else:
                logger.error(f"SQLite error: {e}")
        except Exception as e:
            logger.error(f"Error loading Cursor credentials: {e}")

    def _derive_identifiers(self) -> None:
        """Derive session_id and client_key from access token."""
        if not self._access_token:
            return
        self._session_id = str(uuid.uuid5(_UUID_NAMESPACE, self._access_token))
        self._client_key = hashlib.sha256(self._access_token.encode()).hexdigest()

    def reload_from_sqlite(self) -> None:
        """Reload credentials from SQLite (e.g., after user re-login)."""
        if self._db_path:
            self._load_from_sqlite(self._db_path)
            if self._access_token:
                self._derive_identifiers()

    async def refresh_access_token(self) -> Optional[str]:
        """
        Refresh the access token using the refresh token.

        Endpoint: POST https://api2.cursor.sh/auth/exchange_user_api_key
        Authorization: Bearer {refresh_token}

        Returns:
            New access token, or None if refresh failed
        """
        if not self._refresh_token:
            logger.warning("No refresh token available for token refresh")
            return None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{CURSOR_API_HOST}/auth/exchange_user_api_key",
                    headers={"Authorization": f"Bearer {self._refresh_token}"},
                )
                if response.status_code == 200:
                    data = response.json()
                    new_token = data.get("accessToken") or data.get("token")
                    if new_token:
                        self._access_token = new_token
                        self._derive_identifiers()
                        logger.info("Access token refreshed successfully")
                        return self._access_token
                else:
                    logger.warning(f"Token refresh failed: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Token refresh error: {e}")

        return None

    def get_access_token(self) -> str:
        """
        Returns the JWT bearer token.

        Raises:
            ValueError: If no access token is available
        """
        if not self._access_token:
            # Try reloading from SQLite
            self.reload_from_sqlite()

        if not self._access_token:
            raise ValueError(
                "No Cursor access token available. "
                "Please log in to Cursor IDE or set CURSOR_ACCESS_TOKEN environment variable."
            )
        return self._access_token

    @property
    def machine_id(self) -> str:
        """Machine ID for checksum computation."""
        return self._machine_id or ""

    @property
    def session_id(self) -> str:
        """UUID5 derived from token for x-session-id header."""
        return self._session_id or str(uuid.uuid4())

    @property
    def client_key(self) -> str:
        """SHA256 of token for x-client-key header."""
        return self._client_key or ""

    @property
    def email(self) -> Optional[str]:
        """Cached email address (informational)."""
        return self._email
