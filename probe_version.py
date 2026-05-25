"""
Test different client versions to see if the server rejects 3.2.21.
"""
import asyncio
import httpx
import uuid
from cursor.auth import CursorAuthManager
from cursor.utils import get_cursor_headers
from cursor.config import CURSOR_API_HOST
from cursor.protobuf import (
    encode_string, encode_int32, encode_submessage,
    encode_conversation_message, encode_model_details, encode_explicit_context,
    wrap_connect_envelope, decode_connect_frames
)

async def test_version(version: str):
    auth = CursorAuthManager()
    headers = get_cursor_headers(auth)
    headers["x-cursor-client-version"] = version

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

    try:
        async with httpx.AsyncClient(http2=True, timeout=15) as client:
            response = await client.post(url, content=envelope, headers=headers)
            body = response.content
            frames = decode_connect_frames(body)
            result = "OK"
            for msg_type, pl in frames:
                if msg_type == 2:
                    try:
                        import json
                        data = json.loads(pl)
                        dbg = data.get("error", {}).get("details", [{}])[0].get("debug", {})
                        result = dbg.get("error", "UNKNOWN")
                    except:
                        result = "PARSE_ERROR"
            print(f"Version {version}: {result}")
    except Exception as e:
        print(f"Version {version}: {type(e).__name__}")


async def probe():
    for version in ["0.50.5", "0.50.6", "0.51.0", "3.2.0", "3.2.20", "3.2.21", "3.2.22", "3.3.0", "4.0.0"]:
        await test_version(version)

if __name__ == "__main__":
    asyncio.run(probe())
