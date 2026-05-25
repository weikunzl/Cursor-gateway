"""
Test BiDi endpoint with HTTP/1.1 instead of HTTP/2.
"""
import asyncio
import httpx
import uuid
import json
from cursor.auth import CursorAuthManager
from cursor.utils import get_cursor_headers
from cursor.config import CURSOR_API_HOST
from cursor.protobuf import (
    encode_string, encode_int32, encode_submessage,
    encode_conversation_message, encode_model_details, encode_explicit_context,
    wrap_connect_envelope, decode_connect_frames
)

async def probe():
    auth = CursorAuthManager()
    headers = get_cursor_headers(auth)

    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Hello", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(2, 1)
    inner += encode_submessage(3, encode_explicit_context(""))
    inner += encode_int32(4, 1)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    inner += encode_int32(13, 1)
    inner += encode_int32(19, 1)
    inner += encode_int32(22, 1)
    inner += encode_string(23, convo)
    inner += encode_int32(35, 1)
    inner += encode_int32(46, 1)
    inner += encode_string(54, "Ask")
    payload = encode_submessage(1, inner)
    envelope = wrap_connect_envelope(payload, compress=False)

    url = f"{CURSOR_API_HOST}/aiserver.v1.ChatService/StreamUnifiedChatWithTools"

    print("=== HTTP/1.1 ===")
    try:
        async with httpx.AsyncClient(http2=False, timeout=15) as client:
            response = await client.post(url, content=envelope, headers=headers)
            print(f"Status: {response.status_code}")
            print(f"Body (first 500): {response.text[:500]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    print("\n=== HTTP/2 ===")
    try:
        async with httpx.AsyncClient(http2=True, timeout=15) as client:
            response = await client.post(url, content=envelope, headers=headers)
            print(f"Status: {response.status_code}")
            print(f"Body (first 500): {response.text[:500]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(probe())
