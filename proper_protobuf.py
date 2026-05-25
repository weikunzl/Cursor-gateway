"""
Test with exact encoding from eisbaw/cursor_proper_protobuf.py
"""
import asyncio
import httpx
import uuid
import struct
import gzip
import time
import base64
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from cursor.auth import CursorAuthManager
from cursor.utils import get_cursor_headers
from cursor.config import CURSOR_API_HOST
from cursor.protobuf import decode_connect_frames


class ProtobufEncoder:
    @staticmethod
    def encode_varint(value):
        result = b''
        while value >= 0x80:
            result += bytes([value & 0x7F | 0x80])
            value >>= 7
        result += bytes([value & 0x7F])
        return result

    @staticmethod
    def encode_field(field_num, wire_type, value):
        tag = (field_num << 3) | wire_type
        result = ProtobufEncoder.encode_varint(tag)
        if wire_type == 0:
            result += ProtobufEncoder.encode_varint(value)
        elif wire_type == 2:
            if isinstance(value, str):
                value = value.encode('utf-8')
            result += ProtobufEncoder.encode_varint(len(value)) + value
        elif wire_type == 1:
            result += struct.pack('<Q', value)
        return result


def encode_message(content, role, message_id, chat_mode_enum=None):
    msg = b''
    msg += ProtobufEncoder.encode_field(1, 2, content)
    msg += ProtobufEncoder.encode_field(2, 0, role)
    msg += ProtobufEncoder.encode_field(13, 2, message_id)
    if chat_mode_enum is not None:
        msg += ProtobufEncoder.encode_field(47, 0, chat_mode_enum)
    return msg


def encode_instruction(instruction_text):
    msg = b''
    if instruction_text:
        msg += ProtobufEncoder.encode_field(1, 2, instruction_text)
    return msg


def encode_model(model_name, max_mode=False):
    msg = b''
    msg += ProtobufEncoder.encode_field(1, 2, model_name)
    msg += ProtobufEncoder.encode_field(4, 2, b'')
    if max_mode:
        msg += ProtobufEncoder.encode_field(8, 0, 1)
    return msg


def encode_cursor_setting():
    msg = b''
    msg += ProtobufEncoder.encode_field(1, 2, "cursor\\aisettings")
    msg += ProtobufEncoder.encode_field(3, 2, b'')
    unknown6_msg = b''
    unknown6_msg += ProtobufEncoder.encode_field(1, 2, b'')
    unknown6_msg += ProtobufEncoder.encode_field(2, 2, b'')
    msg += ProtobufEncoder.encode_field(6, 2, unknown6_msg)
    msg += ProtobufEncoder.encode_field(8, 0, 1)
    msg += ProtobufEncoder.encode_field(9, 0, 1)
    return msg


def encode_metadata(client_os, client_arch, client_os_version):
    msg = b''
    msg += ProtobufEncoder.encode_field(1, 2, client_os)
    msg += ProtobufEncoder.encode_field(2, 2, client_arch)
    msg += ProtobufEncoder.encode_field(3, 2, client_os_version)
    msg += ProtobufEncoder.encode_field(4, 2, sys.executable or "python3")
    msg += ProtobufEncoder.encode_field(5, 2, datetime.now().isoformat())
    return msg


def encode_message_id(message_id, role, summary_id=None):
    msg = b''
    msg += ProtobufEncoder.encode_field(1, 2, message_id)
    if summary_id:
        msg += ProtobufEncoder.encode_field(2, 2, summary_id)
    msg += ProtobufEncoder.encode_field(3, 0, role)
    return msg


def encode_request(messages, model_name, client_os, client_arch, client_os_version, max_mode=False):
    msg = b''
    formatted_messages = []
    message_ids = []

    for user_msg in messages:
        if user_msg['role'] == 'user':
            msg_id = str(uuid.uuid4())
            formatted_messages.append({
                'content': user_msg['content'],
                'role': 1,
                'messageId': msg_id,
                'chatModeEnum': 1
            })
            message_ids.append({'messageId': msg_id, 'role': 1})

    for formatted_msg in formatted_messages:
        message_bytes = encode_message(
            formatted_msg['content'],
            formatted_msg['role'],
            formatted_msg['messageId'],
            formatted_msg.get('chatModeEnum')
        )
        msg += ProtobufEncoder.encode_field(1, 2, message_bytes)

    msg += ProtobufEncoder.encode_field(2, 0, 1)
    instruction_bytes = encode_instruction("")
    msg += ProtobufEncoder.encode_field(3, 2, instruction_bytes)
    msg += ProtobufEncoder.encode_field(4, 0, 1)
    model_bytes = encode_model(model_name, max_mode=max_mode)
    msg += ProtobufEncoder.encode_field(5, 2, model_bytes)
    msg += ProtobufEncoder.encode_field(8, 2, "")
    msg += ProtobufEncoder.encode_field(13, 0, 1)
    cursor_setting_bytes = encode_cursor_setting()
    msg += ProtobufEncoder.encode_field(15, 2, cursor_setting_bytes)
    msg += ProtobufEncoder.encode_field(19, 0, 1)
    msg += ProtobufEncoder.encode_field(23, 2, str(uuid.uuid4()))
    metadata_bytes = encode_metadata(client_os, client_arch, client_os_version)
    msg += ProtobufEncoder.encode_field(26, 2, metadata_bytes)
    msg += ProtobufEncoder.encode_field(27, 0, 0)

    for msg_id_data in message_ids:
        message_id_bytes = encode_message_id(
            msg_id_data['messageId'],
            msg_id_data['role']
        )
        msg += ProtobufEncoder.encode_field(30, 2, message_id_bytes)

    msg += ProtobufEncoder.encode_field(35, 0, 0)
    msg += ProtobufEncoder.encode_field(38, 0, 0)
    msg += ProtobufEncoder.encode_field(46, 0, 1)
    msg += ProtobufEncoder.encode_field(47, 2, "")
    msg += ProtobufEncoder.encode_field(48, 0, 0)
    msg += ProtobufEncoder.encode_field(49, 0, 0)
    msg += ProtobufEncoder.encode_field(51, 0, 0)
    msg += ProtobufEncoder.encode_field(53, 0, 1)
    msg += ProtobufEncoder.encode_field(54, 2, "Ask")
    return msg


def encode_stream_unified_chat_request(messages, model_name, client_os, client_arch, client_os_version, max_mode=False):
    msg = b''
    request_bytes = encode_request(messages, model_name, client_os, client_arch, client_os_version, max_mode=max_mode)
    msg += ProtobufEncoder.encode_field(1, 2, request_bytes)
    return msg


def generate_cursor_body_exact(messages, model_name, client_os, client_arch, client_os_version, max_mode=False):
    buffer = encode_stream_unified_chat_request(messages, model_name, client_os, client_arch, client_os_version, max_mode=max_mode)
    magic_number = 0x00
    if len(messages) >= 3:
        buffer = gzip.compress(buffer)
        magic_number = 0x01
    length_hex = format(len(buffer), '08x')
    length_bytes = bytes.fromhex(length_hex)
    final_body = bytes([magic_number]) + length_bytes + buffer
    return final_body


async def test_proper():
    auth = CursorAuthManager()
    token = auth.get_access_token()
    if '::' in token:
        token = token.split('::')[1]

    session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, token))
    client_key = hashlib.sha256(token.encode()).hexdigest()

    machine_id = auth.machine_id or ""
    timestamp = int(time.time() * 1000 // 1000000)
    byte_array = bytearray([
        (timestamp >> 40) & 255, (timestamp >> 32) & 255,
        (timestamp >> 24) & 255, (timestamp >> 16) & 255,
        (timestamp >> 8) & 255, timestamp & 255,
    ])
    t = 165
    for i in range(len(byte_array)):
        byte_array[i] = ((byte_array[i] ^ t) + (i % 256)) & 255
        t = byte_array[i]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    encoded = ""
    for i in range(0, len(byte_array), 3):
        a = byte_array[i]
        b = byte_array[i + 1] if i + 1 < len(byte_array) else 0
        c = byte_array[i + 2] if i + 2 < len(byte_array) else 0
        encoded += alphabet[a >> 2]
        encoded += alphabet[((a & 3) << 4) | (b >> 4)]
        if i + 1 < len(byte_array):
            encoded += alphabet[((b & 15) << 2) | (c >> 6)]
        if i + 2 < len(byte_array):
            encoded += alphabet[c & 63]
    cursor_checksum = f"{encoded}{machine_id}"

    request_id = str(uuid.uuid4())
    client_os = "darwin"
    client_arch = "arm64"
    client_os_version = platform.mac_ver()[0] or platform.release()

    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/connect+proto",
        "connect-protocol-version": "1",
        "user-agent": "connect-es/1.6.1",
        "x-amzn-trace-id": f"Root={request_id}",
        "x-client-key": client_key,
        "x-cursor-checksum": cursor_checksum,
        "x-cursor-client-version": "3.2.21",
        "x-cursor-client-type": "ide",
        "x-cursor-client-os": client_os,
        "x-cursor-client-arch": client_arch,
        "x-cursor-client-os-version": client_os_version,
        "x-cursor-client-device-type": "desktop",
        "x-cursor-config-version": str(uuid.uuid4()),
        "x-cursor-timezone": "Asia/Shanghai",
        "x-ghost-mode": "true",
        "x-new-onboarding-completed": "true",
        "x-request-id": request_id,
        "x-session-id": session_id,
    }

    messages = [{"role": "user", "content": "Hello"}]
    body = generate_cursor_body_exact(messages, "claude-4-opus", client_os, client_arch, client_os_version, max_mode=True)

    url = f"{CURSOR_API_HOST}/aiserver.v1.ChatService/StreamUnifiedChatWithTools"

    print(f"Body size: {len(body)} bytes")
    print(f"Body hex (first 50): {body[:50].hex()}")

    try:
        async with httpx.AsyncClient(http2=True, timeout=30) as client:
            async with client.stream('POST', url, headers=headers, content=body) as response:
                print(f"Status: {response.status_code}")
                full_body = b""
                async for chunk in response.aiter_bytes():
                    full_body += chunk
                print(f"Response size: {len(full_body)} bytes")
                print(f"Response text (first 300): {full_body[:300]}")

                frames = decode_connect_frames(full_body)
                print(f"Decoded frames: {len(frames)}")
                for msg_type, pl in frames:
                    if msg_type == 2:
                        try:
                            data = json.loads(pl)
                            dbg = data.get("error", {}).get("details", [{}])[0].get("debug", {})
                            print(f"Error: {dbg.get('error', 'UNKNOWN')}")
                        except:
                            pass
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(test_proper())
