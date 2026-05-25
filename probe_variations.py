"""
Systematic probe of request variations to isolate the BAD_REQUEST cause.
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
    wrap_connect_envelope, decode_connect_frames, decode_response_proto
)

auth = CursorAuthManager()
headers = get_cursor_headers(auth)
url = f"{CURSOR_API_HOST}/aiserver.v1.ChatService/StreamUnifiedChatWithTools"

async def test(name: str, inner: bytes):
    payload = encode_submessage(1, inner)
    envelope = wrap_connect_envelope(payload, compress=False)
    try:
        async with httpx.AsyncClient(http2=True, timeout=15) as client:
            r = await client.post(url, content=envelope, headers=headers)
            body = r.content
            frames = decode_connect_frames(body)
            result = "OK" if not frames else None
            for msg_type, payload in frames:
                if msg_type == 2:
                    try:
                        data = json.loads(payload)
                        if "error" in data:
                            err = data["error"]
                            dbg = err.get("details", [{}])[0].get("debug", {})
                            result = dbg.get("error", "UNKNOWN")
                    except:
                        result = "JSON_ERR"
            print(f"{name}: {result}")
            return result
    except Exception as e:
        print(f"{name}: {type(e).__name__}")
        return str(type(e).__name__)


def base_inner():
    """Minimal inner request."""
    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Hello", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(22, 1)
    inner += encode_string(23, convo)
    return inner


async def probe():
    print("=== Varying optional fields ===")

    # Base: minimal request
    await test("base", base_inner())

    # With model
    inner = base_inner()
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    await test("+model", inner)

    # With all our usual fields
    inner = base_inner()
    inner += encode_int32(2, 1)
    inner += encode_submessage(3, encode_explicit_context(""))
    inner += encode_int32(4, 1)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    inner += encode_int32(13, 1)
    inner += encode_int32(19, 1)
    inner += encode_int32(35, 1)
    inner += encode_int32(46, 1)
    inner += encode_string(54, "Ask")
    await test("+usual", inner)

    # Without explicit_context
    inner = base_inner()
    inner += encode_int32(2, 1)
    inner += encode_int32(4, 1)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    inner += encode_int32(13, 1)
    inner += encode_int32(19, 1)
    inner += encode_int32(35, 1)
    inner += encode_int32(46, 1)
    inner += encode_string(54, "Ask")
    await test("-explicit_context", inner)

    # Without is_chat
    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Hello", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_string(23, convo)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    await test("-is_chat", inner)

    # is_chat = 0
    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Hello", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(22, 0)
    inner += encode_string(23, convo)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    await test("is_chat=0", inner)

    # Different unified_mode_name values
    for name_val in ["CHAT", "AGENT", "EDIT", "", None]:
        inner = base_inner()
        inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
        inner += encode_int32(46, 1)
        if name_val is not None:
            inner += encode_string(54, name_val)
        await test(f"unified_mode_name={name_val!r}", inner)

    # Different unified_mode values
    for mode_val in [0, 1, 2, 3]:
        inner = base_inner()
        inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
        inner += encode_int32(46, mode_val)
        await test(f"unified_mode={mode_val}", inner)

    # No model at all
    await test("no_model", base_inner())

    # Different models
    for model_name in ["claude-4-opus", "claude-4-sonnet", "claude-3.5-sonnet", "gpt-4o", "cursor-small", "gpt-5.3-codex"]:
        inner = base_inner()
        inner += encode_submessage(5, encode_model_details(model_name, max_mode=(model_name=="claude-4-opus")))
        await test(f"model={model_name}", inner)

    # Empty message text
    convo = str(uuid.uuid4())
    msg = encode_conversation_message("", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(22, 1)
    inner += encode_string(23, convo)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    await test("empty_text", inner)

    # AI type first message
    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Hello", 2, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(22, 1)
    inner += encode_string(23, convo)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    await test("ai_type_first", inner)

    # Without conversation_id
    msg = encode_conversation_message("Hello", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(22, 1)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    await test("-conversation_id", inner)

    # With required-looking fields set
    inner = base_inner()
    inner += encode_int32(25, 0)
    inner += encode_int32(27, 0)
    inner += encode_int32(45, 0)
    inner += encode_int32(67, 0)
    inner += encode_int32(69, 0)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    await test("+required_bools", inner)

    # Without bubble_id
    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Hello", 1, "")
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(22, 1)
    inner += encode_string(23, convo)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    await test("-bubble_id", inner)

    # With should_disable_tools=1
    inner = base_inner()
    inner += encode_int32(48, 1)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    await test("+should_disable_tools", inner)

    # Without any optional flags
    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Hello", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(22, 1)
    inner += encode_string(23, convo)
    inner += encode_submessage(5, encode_model_details("claude-4-opus", max_mode=True))
    await test("minimal+model", inner)


if __name__ == "__main__":
    asyncio.run(probe())
