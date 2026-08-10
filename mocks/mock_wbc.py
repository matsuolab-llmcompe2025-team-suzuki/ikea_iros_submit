#!/usr/bin/env python3
"""Stand-in for the organizer's whole-body controller.

Connects to your client's action socket exactly as the real WBC does,
decodes what you publish, and complains loudly when it does not match the
contract. If this is happy, the real controller will parse your actions.

It cannot tell you whether your actions are *good* — only that they are
well formed. A latent inside the bound can still be nonsense.

    python mocks/mock_wbc.py --lane sonic
    python mocks/mock_wbc.py --lane decoupled --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boundary.actions import (  # noqa: E402
    HEADER_SIZE, LATENT_ABS_BOUND, POSE_TOPIC, TASKSPACE_DIM, TASKSPACE_TOPIC,
)

_TAG_DTYPE = {"f32": np.float32, "f64": np.float64, "i32": np.int32,
              "i64": np.int64, "u8": np.uint8, "bool": np.bool_}


def decode_pose(message: bytes) -> dict[str, np.ndarray]:
    """Unpack a SONIC [topic][1280B JSON header][payload] frame."""
    if not message.startswith(POSE_TOPIC):
        raise ValueError(f"frame does not start with topic {POSE_TOPIC!r}")
    start = len(POSE_TOPIC)
    header = json.loads(message[start:start + HEADER_SIZE].rstrip(b"\x00").decode("utf-8"))
    payload = message[start + HEADER_SIZE:]

    out, offset = {}, 0
    for field in header["fields"]:
        dtype = np.dtype(_TAG_DTYPE[field["dtype"]]).newbyteorder("<")
        count = int(np.prod(field["shape"])) if field["shape"] else 1
        nbytes = dtype.itemsize * count
        out[field["name"]] = np.frombuffer(
            payload[offset:offset + nbytes], dtype=dtype
        ).reshape(field["shape"])
        offset += nbytes
    return out


def decode_taskspace(message: bytes) -> tuple[np.ndarray, float]:
    """Unpack a task-space chunk; returns (T, 25) float32 and issue time."""
    if not message.startswith(TASKSPACE_TOPIC):
        raise ValueError(f"frame does not start with topic {TASKSPACE_TOPIC!r}")
    msg = msgpack.unpackb(message[len(TASKSPACE_TOPIC):], raw=False)
    arr = np.frombuffer(msg["actions"], dtype=np.float32).reshape(msg["shape"])
    return arr, float(msg.get("issued_at", 0.0))


def check_pose(fields: dict[str, np.ndarray]) -> list[str]:
    """Contract checks the real deploy would apply to one latent row."""
    problems = []
    for name, shape in (("token_state", (1, 64)), ("frame_index", (1,)),
                        ("left_hand_joints", (1, 7)), ("right_hand_joints", (1, 7))):
        if name not in fields:
            problems.append(f"missing field {name!r}")
        elif fields[name].shape != shape:
            problems.append(f"{name} has shape {fields[name].shape}, expected {shape}")
    if "token_state" in fields:
        peak = float(np.max(np.abs(fields["token_state"])))
        if peak > LATENT_ABS_BOUND:
            problems.append(f"max|motion_token| = {peak:.3f} > {LATENT_ABS_BOUND}")
    return problems


def check_taskspace(arr: np.ndarray) -> list[str]:
    """Contract checks the real adapter would apply to one chunk."""
    problems = []
    if arr.ndim != 2 or arr.shape[1] != TASKSPACE_DIM:
        return [f"chunk has shape {arr.shape}, expected (T, {TASKSPACE_DIM})"]
    if not np.all(np.isfinite(arr)):
        problems.append("chunk contains non-finite values")
    if np.max(np.abs(arr[:, 0:4])) > 1.0 + 1e-3:
        problems.append("hand commands outside [-1, 1]")
    for side, columns in (("left", slice(7, 11)), ("right", slice(14, 18))):
        norms = np.linalg.norm(arr[:, columns], axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-2):
            problems.append(f"{side} end-effector quaternions are not unit length")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", choices=("sonic", "decoupled"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--expect", type=int, default=0,
                        help="Exit 0 after N clean messages (for scripted checks).")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()

    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{args.host}:{args.port}")   # the client binds, we dial in
    print(f"[mock-wbc] lane={args.lane} connected to tcp://{args.host}:{args.port}")

    seen, rejected = 0, 0
    deadline = time.time() + args.timeout_s
    while time.time() < deadline:
        try:
            blob = socket.recv()
        except zmq.Again:
            continue
        try:
            if args.lane == "sonic":
                fields = decode_pose(blob)
                problems = check_pose(fields)
                summary = f"frame_index={int(fields['frame_index'][0])}"
            else:
                arr, issued = decode_taskspace(blob)
                problems = check_taskspace(arr)
                summary = f"T={arr.shape[0]} age={time.time() - issued:.3f}s"
        except Exception as exc:            # noqa: BLE001 — report, do not crash
            problems, summary = [f"undecodable frame: {exc}"], ""

        if problems:
            rejected += 1
            for problem in problems:
                print(f"[mock-wbc] ACTION REJECTED: {problem}", file=sys.stderr)
        else:
            seen += 1
            if seen % 25 == 1:
                print(f"[mock-wbc] ok #{seen} {summary}")
            if args.expect and seen >= args.expect:
                break
        deadline = time.time() + args.timeout_s

    print(f"[mock-wbc] {seen} accepted, {rejected} rejected")
    if args.expect:
        sys.exit(0 if seen >= args.expect and rejected == 0 else 1)


if __name__ == "__main__":
    main()
