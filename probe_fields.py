"""
Test adding missing protobuf fields that cursor_proper_protobuf.py sets
and that match the 3.2.21 bundle schema.
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

async def test(name: str, inner: bytes):
    auth = CursorAuthManager()
    headers = get_cursor_headers(auth)
    payload = encode_submessage(1, inner)
    envelope = wrap_connect_envelope(payload, compress=False)
    url = f"{CURSOR_API_HOST}/aiserver.v1.ChatService/StreamUnifiedChatWithTools"

    try:
        async with httpx.AsyncClient(http2=True, timeout=15) as client:
            r = await client.post(url, content=envelope, headers=headers)
            frames = decode_connect_frames(r.content)
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


def base_inner():
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
    return inner


async def probe():
    # Baseline
    await test("baseline", base_inner())

    # Add field 8 = "" (use_web)
    inner = base_inner()
    inner += encode_string(8, "")
    await test("+use_web_empty", inner)

    # Add field 27 = 0 (is_agentic)
    inner = base_inner()
    inner += encode_int32(27, 0)
    await test("+is_agentic=0", inner)

    # Add field 38 = 0 (number_of_times_shown_fallback_model_warning)
    inner = base_inner()
    inner += encode_int32(38, 0)
    await test("+fallback_warning=0", inner)

    # Add field 48 = 0 (should_disable_tools)
    inner = base_inner()
    inner += encode_int32(48, 0)
    await test("+should_disable_tools=0", inner)

    # Add field 49 = 0 (thinking_level=UNSPECIFIED)
    inner = base_inner()
    inner += encode_int32(49, 0)
    await test("+thinking_level=0", inner)

    # Add field 51 = 0 (uses_rules)
    inner = base_inner()
    inner += encode_int32(51, 0)
    await test("+uses_rules=0", inner)

    # Add field 53 = 1 (mode_uses_auto_apply)
    inner = base_inner()
    inner += encode_int32(53, 1)
    await test("+mode_uses_auto_apply=1", inner)

    # Add all matching fields at once
    inner = base_inner()
    inner += encode_string(8, "")
    inner += encode_int32(27, 0)
    inner += encode_int32(38, 0)
    inner += encode_int32(48, 0)
    inner += encode_int32(49, 0)
    inner += encode_int32(51, 0)
    inner += encode_int32(53, 1)
    await test("+all_matching_fields", inner)

    # Try with claude-4-sonnet (no max_mode)
    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Hello", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_int32(2, 1)
    inner += encode_submessage(3, encode_explicit_context(""))
    inner += encode_int32(4, 1)
    inner += encode_submessage(5, encode_model_details("claude-4-sonnet", max_mode=False))
    inner += encode_int32(13, 1)
    inner += encode_int32(19, 1)
    inner += encode_int32(22, 1)
    inner += encode_string(23, convo)
    inner += encode_int32(35, 1)
    inner += encode_int32(46, 1)
    inner += encode_string(54, "Ask")
    inner += encode_string(8, "")
    inner += encode_int32(27, 0)
    inner += encode_int32(38, 0)
    inner += encode_int32(48, 0)
    inner += encode_int32(49, 0)
    inner += encode_int32(51, 0)
    inner += encode_int32(53, 1)
    await test("+all_fields_sonnet", inner)

if __name__ == "__main__":
    asyncio.run(probe())
