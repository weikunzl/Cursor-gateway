"""
HTTP client for Copilot API with retry logic.
"""

import asyncio
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from copilot.config import (
    MAX_RETRIES,
    BASE_RETRY_DELAY,
    FIRST_TOKEN_MAX_RETRIES,
    STREAMING_READ_TIMEOUT,
)
from copilot.auth import CopilotAuthManager
from copilot.utils import get_copilot_headers


class CopilotHttpClient:
    def __init__(
        self,
        auth_manager: CopilotAuthManager,
        shared_client: Optional[httpx.AsyncClient] = None,
    ):
        self.auth_manager = auth_manager
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client

    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        if self._shared_client is not None:
            return self._shared_client

        if self.client is None or self.client.is_closed:
            if stream:
                timeout_config = httpx.Timeout(
                    connect=30.0, read=STREAMING_READ_TIMEOUT,
                    write=30.0, pool=30.0,
                )
            else:
                timeout_config = httpx.Timeout(timeout=300.0)

            self.client = httpx.AsyncClient(
                timeout=timeout_config, follow_redirects=True,
            )
        return self.client

    async def close(self) -> None:
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
        data: Dict[str, Any],
        stream: bool = False,
    ) -> httpx.Response:
        max_retries = FIRST_TOKEN_MAX_RETRIES if stream else MAX_RETRIES
        client = await self._get_client(stream=stream)
        last_error = None

        for attempt in range(max_retries):
            try:
                token = await self.auth_manager.get_copilot_token()
                headers = get_copilot_headers(token)

                if stream:
                    req = client.build_request(
                        method, url,
                        json=data,
                        headers=headers,
                    )
                    response = await client.send(req, stream=True)
                else:
                    response = await client.request(
                        method, url,
                        json=data,
                        headers=headers,
                    )

                if response.status_code == 200:
                    return response

                if response.status_code == 401:
                    logger.warning(f"401 from Copilot API, refreshing token (attempt {attempt + 1}/{max_retries})")
                    await self.auth_manager.force_refresh()
                    continue

                if response.status_code == 429:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"429 rate limit, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue

                if 500 <= response.status_code < 600:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"{response.status_code} server error, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue

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

        error_msg = str(last_error) if last_error else "Unknown error"
        status = 504 if stream else 502
        raise HTTPException(
            status_code=status,
            detail=f"Request failed after {max_retries} attempts: {error_msg}",
        )

    async def __aenter__(self) -> "CopilotHttpClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
