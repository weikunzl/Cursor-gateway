"""
ConnectRPC stream parser for Cursor API responses.

Parses the binary envelope framing used by Cursor's ConnectRPC API
and extracts content events from protobuf/JSON frames.
"""

import gzip
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from cursor.protobuf import decode_proto_fields


@dataclass
class StreamEvent:
    """A parsed event from the Cursor stream."""
    type: str  # "content", "thinking", "tool_use", "usage", "end", "error"
    data: Any = None


class ConnectRpcStreamParser:
    """
    Parser for ConnectRPC streaming responses from Cursor API.

    Response uses envelope framing:
    [msg_type:1byte][msg_len:4bytes BE][msg_data]

    msg_type:
    - 0: raw protobuf
    - 1: gzip protobuf
    - 2: raw JSON (end-of-stream or error)
    - 3: gzip JSON
    """

    def __init__(self):
        self._buffer = b""

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        """
        Feed raw bytes from the HTTP response and return parsed events.

        Args:
            chunk: Raw bytes from the response stream

        Returns:
            List of StreamEvent objects
        """
        self._buffer += chunk
        events = []

        while len(self._buffer) >= 5:
            # Read frame header
            msg_type = self._buffer[0]
            msg_len = struct.unpack(">I", self._buffer[1:5])[0]

            # Check if we have the full frame
            if len(self._buffer) < 5 + msg_len:
                break

            # Extract frame payload
            payload = self._buffer[5:5 + msg_len]
            self._buffer = self._buffer[5 + msg_len:]

            # Process frame
            frame_events = self._process_frame(msg_type, payload)
            events.extend(frame_events)

        return events

    def _process_frame(self, msg_type: int, payload: bytes) -> List[StreamEvent]:
        """Process a single frame based on its type."""
        events = []

        # Decompress if needed
        if msg_type == 1:  # gzip protobuf
            try:
                payload = gzip.decompress(payload)
                msg_type = 0
            except Exception as e:
                logger.warning(f"Failed to decompress gzip protobuf: {e}")
                return events
        elif msg_type == 3:  # gzip JSON
            try:
                payload = gzip.decompress(payload)
                msg_type = 2
            except Exception as e:
                logger.warning(f"Failed to decompress gzip JSON: {e}")
                return events

        if msg_type == 0:  # protobuf
            events.extend(self._parse_protobuf_frame(payload))
        elif msg_type == 2:  # JSON (usually end-of-stream or error)
            events.extend(self._parse_json_frame(payload))

        return events

    def _parse_protobuf_frame(self, payload: bytes) -> List[StreamEvent]:
        """Parse a protobuf response frame."""
        events = []

        try:
            fields = decode_proto_fields(payload)

            # Cursor 3.2.21: New wrapper StreamUnifiedChatResponseWithTools
            # Field 2 = stream_unified_chat_response
            # Fallback to Field 1 for backward compatibility
            response_field = 2 if 2 in fields else 1

            if response_field in fields:
                for submsg_bytes in fields[response_field]:
                    if not isinstance(submsg_bytes, bytes):
                        continue

                    sub_fields = decode_proto_fields(submsg_bytes)

                    # Sub-field 1: text content delta
                    if 1 in sub_fields:
                        for val in sub_fields[1]:
                            if isinstance(val, bytes):
                                try:
                                    text = val.decode("utf-8")
                                    if text:
                                        events.append(StreamEvent(type="content", data=text))
                                except UnicodeDecodeError:
                                    pass

                    # Sub-field 25: thinking message (nested)
                    if 25 in sub_fields:
                        for val in sub_fields[25]:
                            if isinstance(val, bytes):
                                try:
                                    thinking_fields = decode_proto_fields(val)
                                    if 1 in thinking_fields:
                                        for tval in thinking_fields[1]:
                                            if isinstance(tval, bytes):
                                                thinking = tval.decode("utf-8")
                                                if thinking:
                                                    events.append(StreamEvent(type="thinking", data=thinking))
                                except UnicodeDecodeError:
                                    pass

            # New wrapper: Field 1 = client_side_tool_v2_call (only when field 2 is the response)
            if response_field == 2 and 1 in fields:
                for submsg_bytes in fields[1]:
                    if isinstance(submsg_bytes, bytes):
                        tool_event = self._parse_tool_call(submsg_bytes)
                        if tool_event:
                            events.append(tool_event)

            # Fallback: old format had tool calls at field 3 in outer wrapper
            elif 3 in fields:
                for submsg_bytes in fields[3]:
                    if isinstance(submsg_bytes, bytes):
                        tool_event = self._parse_tool_call(submsg_bytes)
                        if tool_event:
                            events.append(tool_event)

        except Exception as e:
            logger.debug(f"Error parsing protobuf frame: {e}")

        return events

    def _parse_tool_call(self, data: bytes) -> Optional[StreamEvent]:
        """Parse a tool call from protobuf data."""
        try:
            fields = decode_proto_fields(data)

            tool_call = {}
            # Try to extract tool call fields
            # Field 1: tool name or ID
            if 1 in fields:
                for val in fields[1]:
                    if isinstance(val, bytes):
                        try:
                            tool_call["name"] = val.decode("utf-8")
                        except UnicodeDecodeError:
                            pass

            # Field 2: arguments or content
            if 2 in fields:
                for val in fields[2]:
                    if isinstance(val, bytes):
                        try:
                            tool_call["arguments"] = val.decode("utf-8")
                        except UnicodeDecodeError:
                            pass

            if tool_call:
                return StreamEvent(type="tool_use", data=tool_call)
        except Exception as e:
            logger.debug(f"Error parsing tool call: {e}")

        return None

    def _parse_json_frame(self, payload: bytes) -> List[StreamEvent]:
        """Parse a JSON frame (usually end-of-stream or error)."""
        events = []

        try:
            # Very small JSON payloads (2 bytes) signal end of stream
            if len(payload) <= 2:
                events.append(StreamEvent(type="end"))
                return events

            data = json.loads(payload.decode("utf-8"))

            # Check for error
            if "error" in data:
                events.append(StreamEvent(type="error", data=data["error"]))

            # Check for usage/metadata
            if "messageMetadata" in data:
                metadata = data["messageMetadata"]
                if "usage" in metadata:
                    events.append(StreamEvent(type="usage", data=metadata["usage"]))

            # If it's a small object with no recognized fields, treat as end
            if not events:
                events.append(StreamEvent(type="end"))

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug(f"Error parsing JSON frame: {e}")
            # Treat unparseable JSON as end-of-stream
            events.append(StreamEvent(type="end"))

        return events
