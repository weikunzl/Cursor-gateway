"""
Probe script to find X-Idempotent-Encryption-Key generation.
"""
import asyncio
import httpx
import uuid
import base64
import hashlib
import hmac
from cursor.auth import CursorAuthManager
from cursor.utils import get_cursor_headers
from cursor.config import CURSOR_API_HOST
from cursor.protobuf import encode_string, encode_int32, encode_submessage, wrap_connect_envelope

async def send_probe(name: str, request_data: bytes, url: str, extra_headers: dict = None):
    auth = CursorAuthManager()
    headers = get_cursor_headers(auth)
    headers["x-cursor-client-version"] = "3.2.21"
    if extra_headers:
        headers.update(extra_headers)
    envelope = wrap_connect_envelope(request_data, compress=False)
    try:
        async with httpx.AsyncClient(http2=True, timeout=15) as client:
            r = await client.post(url, content=envelope, headers=headers)
            body = r.text
            print(f"\n=== {name} ===")
            print(f"Status: {r.status_code}")
            print(f"Body: {body[:300]}")
            if "ERROR_BAD_MODEL_NAME" in body:
                return "BAD_MODEL"
            if "ERROR_BAD_REQUEST" in body:
                if "is invalid" in body:
                    return "INVALID_KEY"
                return "BAD_REQUEST"
            if r.status_code == 200 and "error" not in body.lower():
                return "OK"
            return f"HTTP_{r.status_code}"
    except httpx.ReadTimeout:
        return "TIMEOUT"
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def encode_conversation_message(text: str, msg_type: int, bubble_id: str = "") -> bytes:
    result = b""
    result += encode_string(1, text)
    result += encode_int32(2, msg_type)
    if bubble_id:
        result += encode_string(13, bubble_id)
    return result


def encode_model_details(model_name: str) -> bytes:
    result = b""
    result += encode_string(1, model_name)
    return result


async def probe():
    url = f"{CURSOR_API_HOST}/aiserver.v1.ChatService/StreamUnifiedChatWithToolsIdempotent"

    convo = str(uuid.uuid4())
    msg = encode_conversation_message("Hello", 1, str(uuid.uuid4()))
    inner = b""
    inner += encode_submessage(1, msg)
    inner += encode_submessage(5, encode_model_details("claude-4-opus"))
    inner += encode_int32(22, 1)
    inner += encode_string(23, convo)
    tools_request = encode_submessage(1, inner)

    idempotency_key = str(uuid.uuid4())
    idempotent = encode_submessage(1, tools_request)
    idempotent += encode_string(4, idempotency_key)
    idempotent += encode_int32(5, 0)

    auth = CursorAuthManager()
    token = auth.get_access_token()
    machine_id = auth.machine_id or ""

    # Try various key generation methods
    keys = [
        ("hmac_sha256_token", hmac.new(token.encode(), idempotency_key.encode(), hashlib.sha256).hexdigest()),
        ("hmac_sha256_machine", hmac.new(machine_id.encode(), idempotency_key.encode(), hashlib.sha256).hexdigest()),
        ("hmac_sha256_token_machine", hmac.new((token + machine_id).encode(), idempotency_key.encode(), hashlib.sha256).hexdigest()),
        ("hmac_sha256_idem_token", hmac.new(idempotency_key.encode(), token.encode(), hashlib.sha256).hexdigest()),
        ("hmac_sha256_idem_machine", hmac.new(idempotency_key.encode(), machine_id.encode(), hashlib.sha256).hexdigest()),
        ("base64_hmac", base64.b64encode(hmac.new(token.encode(), idempotency_key.encode(), hashlib.sha256).digest()).decode()),
        ("sha256_concat", hashlib.sha256((token + idempotency_key).encode()).hexdigest()),
        ("sha256_concat_rev", hashlib.sha256((idempotency_key + token).encode()).hexdigest()),
        ("md5_concat", hashlib.md5((token + idempotency_key).encode()).hexdigest()),
    ]

    for key_name, encryption_key in keys:
        result = await send_probe(
            f"Idempotent_{key_name}",
            idempotent,
            url,
            {
                "x-idempotency-key": idempotency_key,
                "x-idempotent-encryption-key": encryption_key,
            }
        )
        print(f"{key_name}: {result}")

if __name__ == "__main__":
    asyncio.run(probe())
