"""
Test for non-deterministic responses from Cursor API.
Run many requests with identical payload to check for intermittent success.
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

    results = {}
    for i in range(10):
        try:
            async with httpx.AsyncClient(http2=True, timeout=15) as client:
                response = await client.post(url, content=envelope, headers=headers)
                body = response.content
                frames = decode_connect_frames(body)
                key = "OK"
                for msg_type, pl in frames:
                    if msg_type == 2:
                        try:
                            import json
                            data = json.loads(pl)
                            dbg = data.get("error", {}).get("details", [{}])[0].get("debug", {})
                            key = dbg.get("error", "UNKNOWN")
                        except:
                            key = "JSON_PARSE_ERROR"
                results[key] = results.get(key, 0) + 1
                print(f"Attempt {i+1}: {key}")
        except Exception as e:
            key = type(e).__name__
            results[key] = results.get(key, 0) + 1
            print(f"Attempt {i+1}: {key}")

    print(f"\nSummary after 10 attempts:")
    for k, v in results.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    asyncio.run(probe())
