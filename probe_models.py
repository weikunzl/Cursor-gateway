"""
Test the AvailableModels endpoint to verify headers/auth work.
"""
import asyncio
import httpx
import json
from cursor.auth import CursorAuthManager
from cursor.utils import get_cursor_headers
from cursor.config import CURSOR_API_HOST

async def probe():
    auth = CursorAuthManager()
    headers = get_cursor_headers(auth)

    # AvailableModels is a simple unary RPC
    url = f"{CURSOR_API_HOST}/aiserver.v1.AiService/AvailableModels"

    # Empty protobuf request
    request_data = b""
    envelope = b"\x00\x00\x00\x00\x00" + request_data  # flags=0, len=0

    print(f"Headers: {json.dumps({k: v[:20]+'...' if len(str(v)) > 20 else v for k, v in headers.items()}, indent=2)}")
    print(f"URL: {url}")

    try:
        async with httpx.AsyncClient(http2=True, timeout=30) as client:
            response = await client.post(url, content=envelope, headers=headers)
            print(f"Status: {response.status_code}")
            print(f"Body (first 500): {response.content[:500]}")
            print(f"Body text: {response.text[:500]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(probe())
