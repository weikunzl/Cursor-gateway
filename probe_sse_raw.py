"""
Test SSE endpoint with raw protobuf (no envelope framing).
In Connect protocol, SSE endpoint may accept raw protobuf body.
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
    decode_connect_frames, decode_response_proto
)

async def probe():
    auth = CursorAuthManager()
    headers = get_cursor_headers(auth)

    # Build request
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

    url = f"{CURSOR_API_HOST}/aiserver.v1.ChatService/StreamUnifiedChatWithToolsSSE"

    # Try raw protobuf without envelope
    headers_raw = dict(headers)
    headers_raw["Content-Type"] = "application/proto"
    headers_raw["Accept"] = "text/event-stream"

    print(f"=== SSE with raw protobuf ===")
    print(f"URL: {url}")
    print(f"Payload size: {len(payload)} bytes")
    try:
        async with httpx.AsyncClient(http2=True, timeout=30) as client:
            response = await client.post(url, content=payload, headers=headers_raw)
            print(f"Status: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type')}")
            print(f"Body (first 500): {response.text[:500]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    # Try with Connect+proto envelope
    import struct
    envelope = struct.pack(">BI", 0, len(payload)) + payload
    headers_env = dict(headers)
    headers_env["Content-Type"] = "application/connect+proto"
    headers_env["Accept"] = "application/connect+proto"

    print(f"\n=== SSE with Connect envelope ===")
    try:
        async with httpx.AsyncClient(http2=True, timeout=30) as client:
            response = await client.post(url, content=envelope, headers=headers_env)
            print(f"Status: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type')}")
            print(f"Body (first 500): {response.text[:500]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    # Try with JSON
    json_payload = json.dumps({
        "streamUnifiedChatRequest": {
            "conversation": [{"text": "Hello", "type": 1, "bubbleId": str(uuid.uuid4())}],
            "allowLongFileScan": True,
            "explicitContext": {"context": ""},
            "canHandleFilenamesAfterLanguageIds": True,
            "modelDetails": {"modelName": "claude-4-opus", "maxMode": True},
            "shouldCache": True,
            "useNewCompressionScheme": True,
            "isChat": True,
            "conversationId": convo,
            "useFullInputsContext": True,
            "unifiedMode": 1,
            "unifiedModeName": "Ask"
        }
    })
    headers_json = dict(headers)
    headers_json["Content-Type"] = "application/json"
    headers_json["Accept"] = "text/event-stream"

    print(f"\n=== SSE with JSON body ===")
    try:
        async with httpx.AsyncClient(http2=True, timeout=30) as client:
            response = await client.post(url, content=json_payload, headers=headers_json)
            print(f"Status: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type')}")
            print(f"Body (first 500): {response.text[:500]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(probe())
