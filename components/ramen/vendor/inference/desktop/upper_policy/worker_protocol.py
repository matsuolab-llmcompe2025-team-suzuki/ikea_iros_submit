"""Length-prefixed local IPC used by the policy and hardware processes.

The child process is created locally by the runner and inherits anonymous
pipes.  Pickle is therefore used only across that trusted parent/child
boundary; this module must never be exposed as a network service.
"""

from __future__ import annotations

import pickle
import struct
from typing import Any, BinaryIO


_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("policy worker pipe closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(stream: BinaryIO, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError(f"policy worker message is too large: {len(payload)} bytes")
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def receive_message(stream: BinaryIO) -> Any:
    (size,) = _HEADER.unpack(_read_exact(stream, _HEADER.size))
    if size <= 0 or size > _MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid policy worker message size: {size}")
    return pickle.loads(_read_exact(stream, size))
