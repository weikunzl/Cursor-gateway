"""
Manual protobuf encoding/decoding for Cursor's ConnectRPC API.

Implements just enough protobuf wire format to encode/decode the
StreamUnifiedChatWithTools request/response messages.

Wire format reference: https://protobuf.dev/programming-guides/encoding/
"""

import gzip
import struct
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ==================================================================================================
# Low-level Protobuf Primitives
# ==================================================================================================

def encode_varint(value: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    result = []
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def decode_varint(data: bytes, offset: int) -> Tuple[int, int]:
    """
    Decode a varint from data at the given offset.

    Returns:
        Tuple of (value, new_offset)
    """
    result = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        result |= (byte & 0x7F) << shift
        offset += 1
        if not (byte & 0x80):
            break
        shift += 7
    return result, offset


def encode_field_tag(field_number: int, wire_type: int) -> bytes:
    """Encode a protobuf field tag."""
    return encode_varint((field_number << 3) | wire_type)


def encode_length_delimited(field_number: int, data: bytes) -> bytes:
    """Encode a length-delimited field (wire type 2)."""
    tag = encode_field_tag(field_number, 2)
    length = encode_varint(len(data))
    return tag + length + data


def encode_string(field_number: int, value: str) -> bytes:
    """Encode a string field."""
    if not value:
        return b""
    return encode_length_delimited(field_number, value.encode("utf-8"))


def encode_bytes_field(field_number: int, value: bytes) -> bytes:
    """Encode a bytes field."""
    return encode_length_delimited(field_number, value)


def encode_int32(field_number: int, value: int) -> bytes:
    """Encode an int32 field (wire type 0 = varint)."""
    tag = encode_field_tag(field_number, 0)
    return tag + encode_varint(value)


def encode_submessage(field_number: int, data: bytes) -> bytes:
    """Encode a submessage (embedded message, wire type 2)."""
    return encode_length_delimited(field_number, data)


# ==================================================================================================
# Message Encoding
# ==================================================================================================

def encode_conversation_message(
    text: str, msg_type: int, bubble_id: str = "", chat_mode_enum: int = 1
) -> bytes:
    """
    Encode a ConversationMessage protobuf.

    ConversationMessage {
        string text = 1;
        int32 type = 2;       // 1=HUMAN, 2=AI
        string bubble_id = 13;
        int32 chat_mode_enum = 47;  // 1 = default chat mode
    }
    """
    result = b""
    result += encode_string(1, text)
    result += encode_int32(2, msg_type)
    if bubble_id:
        result += encode_string(13, bubble_id)
    result += encode_int32(47, chat_mode_enum)
    return result


def encode_model_details(model_name: str, max_mode: bool = False) -> bytes:
    """
    Encode a ModelDetails protobuf.

    ModelDetails {
        string model_name = 1;
        bytes unknown_4 = 4;   // always empty
        bool max_mode = 8;
    }
    """
    result = b""
    result += encode_string(1, model_name)
    result += encode_bytes_field(4, b"")
    if max_mode:
        result += encode_int32(8, 1)
    return result


def encode_explicit_context(context: str = "") -> bytes:
    """
    Encode an ExplicitContext protobuf.

    ExplicitContext {
        string context = 1;
    }
    """
    return encode_string(1, context)


def encode_cursor_setting() -> bytes:
    """
    Encode a CursorSetting protobuf (field 15 in request).

    CursorSetting {
        string name = 1;           // "cursor\aisettings"
        bytes unknown_3 = 3;       // empty
        Unknown6 unknown_6 = 6 {
            bytes field_1 = 1;     // empty
            bytes field_2 = 2;     // empty
        }
        bool field_8 = 8;          // 1
        bool field_9 = 9;          // 1
    }
    """
    result = b""
    result += encode_string(1, "cursor\\aisettings")
    result += encode_bytes_field(3, b"")
    unknown6 = b""
    unknown6 += encode_bytes_field(1, b"")
    unknown6 += encode_bytes_field(2, b"")
    result += encode_submessage(6, unknown6)
    result += encode_int32(8, 1)
    result += encode_int32(9, 1)
    return result


def encode_metadata(client_os: str, client_arch: str, client_os_version: str) -> bytes:
    """
    Encode a Metadata protobuf (field 26 in request).

    Metadata {
        string client_os = 1;
        string client_arch = 2;
        string client_os_version = 3;
        string python_executable = 4;
        string timestamp = 5;      // ISO format
    }
    """
    import sys
    from datetime import datetime

    result = b""
    result += encode_string(1, client_os)
    result += encode_string(2, client_arch)
    result += encode_string(3, client_os_version)
    result += encode_string(4, sys.executable or "python3")
    result += encode_string(5, datetime.now().isoformat())
    return result


def encode_message_id(message_id: str, role: int, summary_id: str = "") -> bytes:
    """
    Encode a MessageId protobuf (field 30 in request).

    MessageId {
        string message_id = 1;
        string summary_id = 2;     // optional
        int32 role = 3;
    }
    """
    result = b""
    result += encode_string(1, message_id)
    if summary_id:
        result += encode_string(2, summary_id)
    result += encode_int32(3, role)
    return result


def encode_chat_request(
    messages: List[Dict[str, Any]],
    model: str,
    system_prompt: str = "",
    conversation_id: str = "",
    client_os: str = "darwin",
    client_arch: str = "arm64",
    client_os_version: str = "",
    is_agentic: bool = False,
) -> bytes:
    """
    Encode the full chat request for Cursor's StreamUnifiedChatWithTools RPC.

    Builds:
    StreamUnifiedChatRequestWithTools {
        StreamUnifiedChatRequest stream_unified_chat_request = 1 {
            repeated ConversationMessage conversation = 1;
            bool allow_long_file_scan = 2;              // 1
            ExplicitContext explicit_context = 3;
            bool can_handle_filenames_after_language_ids = 4;  // 1
            ModelDetails model_details = 5;
            string use_web = 8;                         // ""
            bool should_cache = 13;                     // 1
            CursorSetting cursor_setting = 15;
            bool use_new_compression_scheme = 19;       // 1
            string conversation_id = 23;
            Metadata metadata = 26;
            bool is_agentic = 27;                       // 0
            repeated MessageId message_ids = 30;
            bool use_full_inputs_context = 35;          // 0
            int32 fallback_warning_count = 38;          // 0
            int32 unified_mode = 46;                    // 1
            string unknown_47 = 47;                     // ""
            bool should_disable_tools = 48;             // 0
            int32 thinking_level = 49;                  // 0
            bool uses_rules = 51;                       // 0
            bool mode_uses_auto_apply = 53;             // 1
            string unified_mode_name = 54;              // "Ask"
        }
    }

    Args:
        messages: List of dicts with 'role' (str) and 'content' (str).
                  role should be "user" or "assistant".
        model: Model name (e.g., "claude-4-sonnet")
        system_prompt: System prompt text
        conversation_id: Conversation ID (UUID)
        client_os: Client OS string
        client_arch: Client architecture string
        client_os_version: Client OS version string

    Returns:
        Protobuf-encoded bytes for the full request
    """
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    # Build StreamUnifiedChatRequest
    request_data = b""
    message_ids: List[Dict[str, Any]] = []

    # Field 1: repeated ConversationMessage conversation
    role_map = {"user": 1, "assistant": 2, "system": 1}  # system treated as human
    for msg in messages:
        msg_type = role_map.get(msg.get("role", "user"), 1)
        text = msg.get("content", "")
        bubble_id = msg.get("message_id", str(uuid.uuid4()))
        msg_bytes = encode_conversation_message(text, msg_type, bubble_id)
        request_data += encode_submessage(1, msg_bytes)
        message_ids.append({"message_id": bubble_id, "role": msg_type})

    # Field 2: allow_long_file_scan = 1
    request_data += encode_int32(2, 1)

    # Field 3: explicit_context (system prompt, always sent even if empty)
    ctx_bytes = encode_explicit_context(system_prompt)
    request_data += encode_submessage(3, ctx_bytes)

    # Field 4: can_handle_filenames_after_language_ids = 1
    request_data += encode_int32(4, 1)

    # Field 5: model_details
    # Some models (e.g., claude-4-opus) require max_mode to be enabled
    max_mode = model in {"claude-4-opus"}
    model_bytes = encode_model_details(model, max_mode=max_mode)
    request_data += encode_submessage(5, model_bytes)

    # Field 8: use_web = ""
    request_data += encode_string(8, "")

    # Field 13: should_cache = 1
    request_data += encode_int32(13, 1)

    # Field 15: cursor_setting
    request_data += encode_submessage(15, encode_cursor_setting())

    # Field 19: use_new_compression_scheme = 1
    request_data += encode_int32(19, 1)

    # Field 23: conversation_id
    request_data += encode_string(23, conversation_id)

    # Field 26: metadata
    request_data += encode_submessage(26, encode_metadata(client_os, client_arch, client_os_version))

    # Field 27: is_agentic
    request_data += encode_int32(27, 1 if is_agentic else 0)

    # Field 30: message_ids
    for mid in message_ids:
        mid_bytes = encode_message_id(mid["message_id"], mid["role"])
        request_data += encode_submessage(30, mid_bytes)

    # Field 35: use_full_inputs_context = 0
    request_data += encode_int32(35, 0)

    # Field 38: fallback_warning_count = 0
    request_data += encode_int32(38, 0)

    # Field 46: unified_mode = 1
    request_data += encode_int32(46, 1)

    # Field 47: unknown string = ""
    request_data += encode_string(47, "")

    # Field 48: should_disable_tools = 0
    request_data += encode_int32(48, 0)

    # Field 49: thinking_level = 0
    request_data += encode_int32(49, 0)

    # Field 51: uses_rules = 0
    request_data += encode_int32(51, 0)

    # Field 53: mode_uses_auto_apply = 1
    request_data += encode_int32(53, 1)

    # Field 54: unified_mode_name
    request_data += encode_string(54, "Agent" if is_agentic else "Ask")

    # Wrap in StreamUnifiedChatRequestWithTools field 1
    return encode_submessage(1, request_data)


# ==================================================================================================
# ConnectRPC Envelope
# ==================================================================================================

def wrap_connect_envelope(payload: bytes, compress: bool = False) -> bytes:
    """
    Wrap protobuf payload in ConnectRPC envelope.

    Format: [flags:1byte][length:4bytes BE][payload]

    Args:
        payload: Protobuf-encoded bytes
        compress: Whether to gzip-compress the payload

    Returns:
        Envelope-wrapped bytes
    """
    if compress:
        payload = gzip.compress(payload)
        flags = 0x01
    else:
        flags = 0x00

    header = struct.pack(">BI", flags, len(payload))
    return header + payload


# ==================================================================================================
# Response Decoding
# ==================================================================================================

def decode_connect_frames(data: bytes) -> List[Tuple[int, bytes]]:
    """
    Parse ConnectRPC envelope frames from response data.

    Each frame: [msg_type:1byte][msg_len:4bytes BE][msg_data]

    msg_type:
    - 0: raw protobuf
    - 1: gzip-compressed protobuf
    - 2: raw JSON (end-of-stream or error)
    - 3: gzip-compressed JSON

    Returns:
        List of (msg_type, payload) tuples
    """
    frames = []
    offset = 0

    while offset + 5 <= len(data):
        msg_type = data[offset]
        msg_len = struct.unpack(">I", data[offset + 1:offset + 5])[0]
        offset += 5

        if offset + msg_len > len(data):
            break

        payload = data[offset:offset + msg_len]
        offset += msg_len

        # Decompress if needed
        if msg_type == 1:  # gzip protobuf
            try:
                payload = gzip.decompress(payload)
                msg_type = 0  # treat as raw protobuf after decompression
            except Exception:
                pass
        elif msg_type == 3:  # gzip JSON
            try:
                payload = gzip.decompress(payload)
                msg_type = 2  # treat as raw JSON after decompression
            except Exception:
                pass

        frames.append((msg_type, payload))

    return frames


def decode_proto_fields(data: bytes) -> Dict[int, List[Any]]:
    """
    Decode all fields from a protobuf message into a dict.

    Returns:
        Dict mapping field_number to list of values.
        Varint fields (wire type 0) -> int values
        Length-delimited fields (wire type 2) -> bytes values
        Fixed32 (wire type 5) -> 4-byte values
        Fixed64 (wire type 1) -> 8-byte values
    """
    fields: Dict[int, List[Any]] = {}
    offset = 0

    while offset < len(data):
        try:
            tag, offset = decode_varint(data, offset)
        except (IndexError, ValueError):
            break

        field_number = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 0:  # varint
            value, offset = decode_varint(data, offset)
            fields.setdefault(field_number, []).append(value)
        elif wire_type == 2:  # length-delimited
            length, offset = decode_varint(data, offset)
            if offset + length > len(data):
                break
            value = data[offset:offset + length]
            offset += length
            fields.setdefault(field_number, []).append(value)
        elif wire_type == 5:  # fixed32
            if offset + 4 > len(data):
                break
            value = data[offset:offset + 4]
            offset += 4
            fields.setdefault(field_number, []).append(value)
        elif wire_type == 1:  # fixed64
            if offset + 8 > len(data):
                break
            value = data[offset:offset + 8]
            offset += 8
            fields.setdefault(field_number, []).append(value)
        else:
            # Unknown wire type, skip
            break

    return fields


def decode_response_proto(data: bytes) -> Dict[str, Any]:
    """
    Decode a Cursor streaming response protobuf message.

    The response structure (Cursor 3.2.21):
    StreamUnifiedChatResponseWithTools {
        - Field 1 (submessage): client_side_tool_v2_call
        - Field 2 (submessage): stream_unified_chat_response
          - Field 1 (string): text content delta
          - Field 25 (message): thinking
            - Field 1 (string): thinking text
        - Field 3 (submessage): conversation_summary
    }

    Returns:
        Dict with optional keys:
        - 'text': content text delta
        - 'thinking': thinking/reasoning text delta
        - 'tool_call': tool call data dict
        - 'raw_fields': all decoded fields (for debugging)
    """
    result: Dict[str, Any] = {}

    fields = decode_proto_fields(data)

    # New wrapper: Field 2 = stream_unified_chat_response
    # Fallback to Field 1 for backward compatibility with old format
    response_field = 2 if 2 in fields else 1

    if response_field in fields:
        for submsg_bytes in fields[response_field]:
            if isinstance(submsg_bytes, bytes):
                sub_fields = decode_proto_fields(submsg_bytes)

                # Sub-field 1: text content delta
                if 1 in sub_fields:
                    for val in sub_fields[1]:
                        if isinstance(val, bytes):
                            try:
                                text = val.decode("utf-8")
                                result["text"] = result.get("text", "") + text
                            except UnicodeDecodeError:
                                pass

                # Sub-field 25: thinking message
                if 25 in sub_fields:
                    for val in sub_fields[25]:
                        if isinstance(val, bytes):
                            try:
                                thinking_fields = decode_proto_fields(val)
                                if 1 in thinking_fields:
                                    for tval in thinking_fields[1]:
                                        if isinstance(tval, bytes):
                                            thinking = tval.decode("utf-8")
                                            result["thinking"] = result.get("thinking", "") + thinking
                            except UnicodeDecodeError:
                                pass

    # Field 1 in new wrapper = client_side_tool_v2_call (if field 2 was used)
    if response_field == 2 and 1 in fields:
        for submsg_bytes in fields[1]:
            if isinstance(submsg_bytes, bytes):
                result["tool_call_raw"] = submsg_bytes

    return result
