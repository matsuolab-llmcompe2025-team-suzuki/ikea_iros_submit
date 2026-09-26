#!/usr/bin/env python3
"""Observe the local-only mock test's joint wire; never publish commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

import msgpack
import numpy as np
import zmq


def decode_joint(message: bytes) -> tuple[np.ndarray, float]:
    if not message.startswith(b"joint"):
        raise ValueError("Expected joint topic")
    payload = msgpack.unpackb(message[5:], raw=False)
    if payload["dtype"] != "f32":
        raise ValueError("Expected f32")
    rows = np.frombuffer(payload["actions"], dtype=np.float32).reshape(payload["shape"])
    if rows.ndim != 2 or rows.shape[1] != 22 or not 1 <= len(rows) <= 64:
        raise ValueError(f"Invalid joint shape: {rows.shape}")
    if not np.isfinite(rows).all() or np.max(np.abs(rows[:, :4])) > 1.001:
        raise ValueError("Invalid joint values or hand range")
    if not np.allclose(rows[:, 21], 0.74, atol=1e-6):
        raise ValueError("Expected configured base height 0.74 m")
    issued_at = float(payload["issued_at"])
    if not np.isfinite(issued_at) or issued_at <= 0:
        raise ValueError("Invalid issued_at")
    return rows.copy(), issued_at


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect("tcp://127.0.0.1:5556")
    rows, issued, received, lengths, errors = [], [], [], [], []
    start = time.monotonic()
    try:
        while not stop and time.monotonic() - start < args.seconds:
            if not socket.poll(100):
                continue
            message = socket.recv()
            try:
                chunk, stamp = decode_joint(message)
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
                continue
            rows.append(chunk)
            issued.append(stamp)
            received.append(time.monotonic())
            lengths.append(len(chunk))
    finally:
        socket.close()
        context.term()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"),
                        rows=np.concatenate(rows) if rows else np.empty((0, 22)),
                        issued_at=issued, received_monotonic=received, chunk_lengths=lengths)
    delta = np.diff(received)
    result = {"passed": bool(rows) and not errors, "messages": len(rows), "errors": errors,
              "endpoint": "tcp://127.0.0.1:5556", "physical_commands_sent": False,
              "interval_ms_p50_p95_max": (np.percentile(delta * 1000, [50, 95, 100]).tolist()
                                          if len(delta) else None)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
