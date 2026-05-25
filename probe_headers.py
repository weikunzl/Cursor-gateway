"""
Test if missing headers cause ERROR_BAD_REQUEST.
"""
import asyncio
import httpx
import uuid
import platform
from cursor.auth import CursorAuthManager
from cursor.utils import get_cursor_headers
from cursor.config import CURSOR_API_HOST
from cursor.protobuf import (
    encode_string, encode_int32, encode_submessage,
    encode_conversation_message, encode_model_details, encode_explicit_context,
    wrap_connect_envelope, decode_connect_frames
)

async def test_with_headers(name: str, extra_headers: dict):
    auth = CursorAuthManager()
    headers = get_cursor_headers(auth)
    headers.update(extra_headers)

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
            print(f"{name}: {result}")
            return result
    except Exception as e:
        print(f"{name}: {type(e).__name__}")
        return str(type(e).__name__)


async def probe():
    req_id = str(uuid.uuid4())
    os_version = platform.mac_ver()[0] or platform.release()

    # Baseline
    await test_with_headers("baseline", {})

    # Add x-cursor-config-version
    await test_with_headers("+config_version", {"x-cursor-config-version": str(uuid.uuid4())})

    # Add x-cursor-timezone
    await test_with_headers("+timezone", {"x-cursor-timezone": "Asia/Shanghai"})

    # Add x-amzn-trace-id
    await test_with_headers("+amzn_trace", {"x-amzn-trace-id": f"Root={req_id}"})

    # Add x-cursor-client-os-version
    await test_with_headers("+os_version", {"x-cursor-client-os-version": os_version})

    # Add x-new-onboarding-completed
    await test_with_headers("+onboarding", {"x-new-onboarding-completed": "true"})

    # Add ALL missing headers at once
    await test_with_headers("+all_headers", {
        "x-cursor-config-version": str(uuid.uuid4()),
        "x-cursor-timezone": "Asia/Shanghai",
        "x-amzn-trace-id": f"Root={req_id}",
        "x-cursor-client-os-version": os_version,
        "x-new-onboarding-completed": "true",
    })

    # Try with all headers AND old client version
    await test_with_headers("+all_headers_v0.50.5", {
        "x-cursor-client-version": "0.50.5",
        "x-cursor-config-version": str(uuid.uuid4()),
        "x-cursor-timezone": "Asia/Shanghai",
        "x-amzn-trace-id": f"Root={req_id}",
        "x-cursor-client-os-version": os_version,
        "x-new-onboarding-completed": "true",
    })

if __name__ == "__main__":
    asyncio.run(probe())
