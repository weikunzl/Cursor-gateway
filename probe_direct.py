"""
Direct probe to Cursor API with raw response dumping.
Tests if claude-4-opus returns non-deterministic responses.
"""
import asyncio
import httpx
import uuid
import json
from cursor.auth import CursorAuthManager
from cursor.utils import get_cursor_headers
from cursor.config import CURSOR_API_HOST
from cursor.protobuf import encode_chat_request, wrap_connect_envelope, decode_connect_frames, decode_response_proto

async def probe():
    auth = CursorAuthManager()
    headers = get_cursor_headers(auth)

    # Build request exactly like the gateway does
    messages = [{"role": "user", "content": "Say hello"}]
    request_data = encode_chat_request(
        messages=messages,
        model="claude-4-opus",
        system_prompt="",
        conversation_id=str(uuid.uuid4()),
    )
    envelope = wrap_connect_envelope(request_data, compress=False)

    url = f"{CURSOR_API_HOST}/aiserver.v1.ChatService/StreamUnifiedChatWithTools"

    print(f"Headers: {json.dumps({k: v[:20]+'...' if len(str(v)) > 20 else v for k, v in headers.items()}, indent=2)}")
    print(f"Request envelope size: {len(envelope)} bytes")
    print(f"URL: {url}")
    print()

    for attempt in range(3):
        print(f"=== Attempt {attempt + 1} ===")
        try:
            async with httpx.AsyncClient(http2=True, timeout=30) as client:
                response = await client.post(url, content=envelope, headers=headers)
                print(f"Status: {response.status_code}")
                print(f"Headers: {dict(response.headers)}")

                body = response.content
                print(f"Body size: {len(body)} bytes")
                print(f"Body hex (first 100): {body[:100].hex()}")
                print(f"Body text (first 200): {body[:200]}")

                if body:
                    frames = decode_connect_frames(body)
                    print(f"Decoded frames: {len(frames)}")
                    for i, (msg_type, payload) in enumerate(frames):
                        print(f"  Frame {i}: type={msg_type}, len={len(payload)}")
                        if msg_type == 0:  # protobuf
                            decoded = decode_response_proto(payload)
                            print(f"    Decoded: {decoded}")
                        elif msg_type == 2:  # JSON
                            try:
                                print(f"    JSON: {json.loads(payload)}")
                            except:
                                print(f"    Raw: {payload}")
                print()
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            print()

if __name__ == "__main__":
    asyncio.run(probe())
