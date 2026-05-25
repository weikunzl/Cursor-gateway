"""
HTTP client for Cursor API with retry logic.

Handles:
- 401: reload token from SQLite and retry
- 429: exponential backoff
- 5xx: exponential backoff
- Timeouts: exponential backoff

Requires HTTP/2 for ConnectRPC protocol.
"""

import asyncio
from typing import Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from cursor.config import MAX_RETRIES, BASE_RETRY_DELAY, FIRST_TOKEN_MAX_RETRIES, STREAMING_READ_TIMEOUT
from cursor.auth import CursorAuthManager
from cursor.utils import get_cursor_headers


class CursorHttpClient:
    """
    HTTP/2 client for Cursor API with retry logic.

    Supports shared client mode for connection pooling.
    """

    def __init__(
        self,
        auth_manager: CursorAuthManager,
        shared_client: Optional[httpx.AsyncClient] = None,
    ):
        self.auth_manager = auth_manager
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client

    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        """Returns or creates an HTTP/2 client."""
        if self._shared_client is not None:
            return self._shared_client

        if self.client is None or self.client.is_closed:
            if stream:
                timeout_config = httpx.Timeout(
                    connect=30.0,
                    read=STREAMING_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0,
                )
            else:
                timeout_config = httpx.Timeout(timeout=300.0)

            self.client = httpx.AsyncClient(
                http2=True,
                timeout=timeout_config,
                follow_redirects=True,
            )
        return self.client

    async def close(self) -> None:
        """Closes the HTTP client if owned."""
        if not self._owns_client:
            return
        if self.client and not self.client.is_closed:
            try:
                await self.client.aclose()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")

    async def request_with_retry(
        self,
        method: str,
        url: str,
        data: bytes,
        stream: bool = False,
    ) -> httpx.Response:
        """
        Executes an HTTP request with retry logic.

        Args:
            method: HTTP method
            url: Request URL
            data: Protobuf-encoded request body (bytes)
            stream: Use streaming mode

        Returns:
            httpx.Response

        Raises:
            HTTPException: On failure after all attempts
        """
        max_retries = FIRST_TOKEN_MAX_RETRIES if stream else MAX_RETRIES
        client = await self._get_client(stream=stream)
        last_error = None

        for attempt in range(max_retries):
            try:
                headers = get_cursor_headers(self.auth_manager)

                if stream:
                    headers["Connection"] = "close"
                    req = client.build_request(
                        method, url,
                        content=data,
                        headers=headers,
                    )
                    logger.debug("Sending streaming request to Cursor API...")
                    response = await client.send(req, stream=True)
                else:
                    logger.debug("Sending request to Cursor API...")
                    response = await client.request(
                        method, url,
                        content=data,
                        headers=headers,
                    )

                if response.status_code == 200:
                    return response

                # 401 - token expired, reload and retry
                if response.status_code == 401:
                    logger.warning(f"Received 401, reloading token (attempt {attempt + 1}/{max_retries})")
                    self.auth_manager.reload_from_sqlite()
                    # Try async refresh if available
                    new_token = await self.auth_manager.refresh_access_token()
                    if not new_token:
                        logger.warning("Token refresh failed, using reloaded token")
                    continue

                # 429 - rate limit
                if response.status_code == 429:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Received 429, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue

                # 5xx - server error
                if 500 <= response.status_code < 600:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Received {response.status_code}, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue

                # Other errors - return as is
                return response

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Timeout: {e} - waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Timeout: {e} - no more retries")

            except httpx.RequestError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Request error: {e} - waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request error: {e} - no more retries")

        # All attempts exhausted
        error_msg = str(last_error) if last_error else "Unknown error"
        status = 504 if stream else 502
        raise HTTPException(
            status_code=status,
            detail=f"Request failed after {max_retries} attempts: {error_msg}"
        )

    async def __aenter__(self) -> "CursorHttpClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
