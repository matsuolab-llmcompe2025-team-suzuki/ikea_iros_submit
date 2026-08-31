"""Small, local-only NumPy protocol for the isolated GR00T runtime."""

from __future__ import annotations

import io
import socket
import struct
from typing import Any

import numpy as np


MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _recv_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_archive(connection: socket.socket) -> dict[str, np.ndarray] | None:
    header = _recv_exact(connection, 8)
    if header is None:
        return None
    size = struct.unpack("!Q", header)[0]
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid GR00T worker message size: {size}")
    payload = _recv_exact(connection, size)
    if payload is None:
        return None
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def send_archive(connection: socket.socket, **arrays: Any) -> None:
    output = io.BytesIO()
    np.savez(output, **arrays)
    payload = output.getvalue()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"GR00T worker response is too large: {len(payload)} bytes")
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


def scalar_text(value: np.ndarray, name: str) -> str:
    if value.size != 1:
        raise ValueError(f"{name} must contain one string")
    return str(value.reshape(-1)[0])
