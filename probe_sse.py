"""
Probe Cursor API via SSE endpoint and with additional required-looking fields.
"""
import asyncio
import httpx
import uuid
import json
from cursor.auth import CursorAuthManager
from cursor.utils import get_cursor_headers
from cursor.config import CURSOR_API_HOST
from cursor.protobuf import (
    encode_string, encode_int32, encode_submessage, encode_conversation_message,
    encode_model_details, encode_explicit_context,
    wrap_connect_envelope, decode_connect_frames, decode_response_proto
)

async def send_request(name: str, request_data: bytes, url_path: str):
    auth = CursorAuthManager()
    headers = get_cursor_headers(auth)
    envelope = wrap_connect_envelope(request_data, compress=False)
    url = f"{CURSOR_API_HOST}{url_path}"

    print(f"\n=== {name} ===")
    print(f"URL: {url}")
    try:
        async with httpx.AsyncClient(http2=True, timeout=30) as client:
            response = await client.post(url, content=envelope, headers=headers)
            print(f"Status: {response.status_code}")
            body = response.content
            print(f"Body size: {len(body)} bytes")
            print(f"Body text (first 300): {body[:300]}")

            frames = decode_connect_frames(body)
            print(f"Decoded frames: {len(frames)}")
            for i, (msg_type, payload) in enumerate(frames):
                if msg_type == 0:
                    decoded = decode_response_proto(payload)
                    print(f"  Frame {i}: protobuf: {decoded}")
                elif msg_type == 2:
                    try:
                        print(f"  Frame {i}: JSON: {json.loads(payload)}")
                    except:
                        print(f"  Frame {i}: raw: {payload[:100]}")
            return response.status_code, body
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return None, None


def build_request_basic(model="claude-4-opus", max_mode=True):
    """Build basic request like gateway currently does."""
    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Say hello", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(2, 1)
    inner += encode_submessage(3, encode_explicit_context(""))
    inner += encode_int32(4, 1)
    inner += encode_submessage(5, encode_model_details(model, max_mode=max_mode))
    inner += encode_int32(13, 1)
    inner += encode_int32(19, 1)
    inner += encode_int32(22, 1)
    inner += encode_string(23, convo)
    inner += encode_int32(35, 1)
    inner += encode_int32(46, 1)
    inner += encode_string(54, "Ask")
    return encode_submessage(1, inner)


def build_request_full(model="claude-4-opus", max_mode=True):
    """Build request with all non-optional-looking fields set."""
    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Say hello", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(2, 1)
    inner += encode_submessage(3, encode_explicit_context(""))
    inner += encode_int32(4, 1)
    inner += encode_submessage(5, encode_model_details(model, max_mode=max_mode))
    inner += encode_int32(13, 1)
    inner += encode_int32(19, 1)
    inner += encode_int32(22, 1)
    inner += encode_string(23, convo)
    inner += encode_int32(25, 0)  # repository_info_should_query_staging = false
    inner += encode_int32(27, 0)  # is_agentic = false
    inner += encode_int32(35, 1)
    inner += encode_int32(45, 0)  # is_headless = false
    inner += encode_int32(46, 1)
    inner += encode_int32(67, 0)  # supports_git_index = false
    inner += encode_int32(69, 0)  # force_is_not_dev = false
    inner += encode_string(54, "Ask")
    return encode_submessage(1, inner)


async def probe():
    # Test 1: BiDi with basic request
    await send_request(
        "BiDi basic claude-4-opus",
        build_request_basic("claude-4-opus", max_mode=True),
        "/aiserver.v1.ChatService/StreamUnifiedChatWithTools"
    )

    # Test 2: BiDi with full request
    await send_request(
        "BiDi full claude-4-opus",
        build_request_full("claude-4-opus", max_mode=True),
        "/aiserver.v1.ChatService/StreamUnifiedChatWithTools"
    )

    # Test 3: SSE with basic request
    await send_request(
        "SSE basic claude-4-opus",
        build_request_basic("claude-4-opus", max_mode=True),
        "/aiserver.v1.ChatService/StreamUnifiedChatWithToolsSSE"
    )

    # Test 4: SSE with basic claude-4-sonnet
    await send_request(
        "SSE basic claude-4-sonnet",
        build_request_basic("claude-4-sonnet", max_mode=False),
        "/aiserver.v1.ChatService/StreamUnifiedChatWithToolsSSE"
    )

    # Test 5: Poll with basic request
    await send_request(
        "Poll basic claude-4-opus",
        build_request_basic("claude-4-opus", max_mode=True),
        "/aiserver.v1.ChatService/StreamUnifiedChatWithToolsPoll"
    )


if __name__ == "__main__":
    asyncio.run(probe())
