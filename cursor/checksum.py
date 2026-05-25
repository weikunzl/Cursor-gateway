"""
Cursor checksum computation for x-cursor-checksum header.

The checksum is a timestamp-based XOR chain encoded in base64url,
concatenated with the machine ID.
"""

import base64
import time


def compute_checksum(machine_id: str) -> str:
    """
    Computes the x-cursor-checksum header value.

    Algorithm:
    1. Derive 6-byte array from current timestamp
    2. XOR-chain with initial key t=165
    3. Base64url encode (no padding) + machine_id

    Args:
        machine_id: Machine ID from Cursor's state.vscdb

    Returns:
        Checksum string for the x-cursor-checksum header
    """
    timestamp = int(time.time() * 1000 // 1000000)
    byte_array = [
        (timestamp >> 40) & 255,
        (timestamp >> 32) & 255,
        (timestamp >> 24) & 255,
        (timestamp >> 16) & 255,
        (timestamp >> 8) & 255,
        timestamp & 255,
    ]
    t = 165
    for i in range(6):
        byte_array[i] = ((byte_array[i] ^ t) + i) % 256
        t = byte_array[i]

    encoded = base64.urlsafe_b64encode(bytes(byte_array)).rstrip(b"=").decode()
    return encoded + machine_id
