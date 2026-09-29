#!/usr/bin/env python3
"""Replay captured packets into the pinned organizer's fake WBC/Dex1 backend.

No sockets or robot SDK are used. Measured state is seeded at the packet's first
target to test mapping without physical tracking error. Timestamp freshness is
tested separately: archived stamps are replaced only after the official decoder
has accepted the original envelope.
"""

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

ORGANIZER_REVISION = "497f3ab93e5baa706311daebd31c7a9798258450"


def read_packets(path):
    with path.open("rb") as source:
        while header := source.read(4):
            if len(header) != 4:
                raise ValueError("Truncated packet header")
            length = struct.unpack("!I", header)[0]
            if not 1 <= length <= 1_000_000:
                raise ValueError("Invalid packet length")
            packet = source.read(length)
            if len(packet) != length:
                raise ValueError("Truncated packet")
            yield packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organizer", type=Path, required=True)
    capture = parser.add_mutually_exclusive_group(required=True)
    capture.add_argument("--capture", type=Path)
    capture.add_argument("--rows-capture", type=Path,
                         help="Legacy decoded NPZ: tests rows, not the original envelope")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.organizer), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != ORGANIZER_REVISION:
        raise ValueError("Unexpected organizer revision")
    sys.path.insert(0, str(args.organizer / "reference/wbc_adapter/tests"))
    import test_joint_lane as official
    from wire_probe import decode_joint, decode_goto

    dex1 = official.FakeDex1()
    ctx = official.make_ctx(dex1=dex1)
    count = 0
    goto_count = 0

    def legacy_packets():
        with np.load(args.rows_capture, allow_pickle=False) as data:
            rows, lengths, stamps = data["rows"], data["chunk_lengths"], data["issued_at"]
            if len(lengths) != len(stamps) or int(lengths.sum()) != len(rows):
                raise ValueError("Inconsistent archived row counts")
            offset = 0
            for length, stamp in zip(lengths, stamps):
                length = int(length)
                if not 1 <= length <= 64:
                    raise ValueError("Invalid archived chunk length")
                yield official.joint_msg(rows[offset:offset + length], issued_at=float(stamp))
                offset += length

    packets = read_packets(args.capture) if args.capture else legacy_packets()
    for packet in packets:
        if packet.startswith(b"goto"):
            target, speed, stamp = decode_goto(packet)
            official.boundary_wire.decode_goto(packet)
            ctx.backend.goals.clear()
            dex1.sent.clear()
            official.wbc_driver._handle_goto(
                ctx, official.goto_msg(target[:7], target[7:], max_speed=speed, issued_at=time.time())
            )
            if len(ctx.backend.goals) != 1 or dex1.sent:
                raise AssertionError("Goto was not accepted once with hands preserved")
            goal = ctx.backend.goals[0]
            upper = np.asarray(goal["target_upper_body_pose"])
            actual = np.c_[upper[:, :7], upper[:, 14:21]]
            np.testing.assert_allclose(actual[-1], target, atol=1e-6, rtol=0)
            if len(actual) > 1 and np.max(np.abs(np.diff(actual, axis=0))) > speed / ctx.args.chunk_hz + 1e-6:
                raise AssertionError("Goto velocity exceeded requested bound")
            np.testing.assert_allclose(goal["navigate_cmd"], 0.0, atol=1e-6)
            if len(actual) / ctx.args.chunk_hz > 15:
                raise AssertionError("Goto duration exceeded organizer limit")
            # Offline fake arrival, not a claim about the captured robot's motion.
            ctx.backend.q[15:22] = target[:7]
            ctx.backend.q[29:36] = target[7:]
            goto_count += 1
            continue
        rows, stamp = decode_joint(packet)
        if stamp <= 0:
            raise ValueError("Invalid original timestamp")
        # Validate the actual transmitted envelope before refreshing its age.
        decoded = official.boundary_wire.decode_joint(packet)
        np.testing.assert_array_equal(decoded.actions, rows)
        ctx.backend.q[15:22] = rows[0, 4:11]
        ctx.backend.q[29:36] = rows[0, 11:18]
        ctx.backend.goals.clear()
        dex1.sent.clear()
        official.wbc_driver._handle_joint(ctx, official.joint_msg(rows, issued_at=time.time()))
        if len(ctx.backend.goals) != 1 or len(dex1.sent) != 1:
            raise AssertionError("Packet was not relayed exactly once")
        goal = ctx.backend.goals[0]
        upper = np.asarray(goal["target_upper_body_pose"])
        np.testing.assert_allclose(upper[:, :7], rows[:, 4:11], atol=1e-6, rtol=0)
        np.testing.assert_allclose(upper[:, 14:21], rows[:, 11:18], atol=1e-6, rtol=0)
        np.testing.assert_allclose(goal["navigate_cmd"], rows[:, 18:21], atol=1e-6, rtol=0)
        np.testing.assert_allclose(goal["base_height_command"], rows[:, 21:22], atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            [dex1.sent[0]["left"], dex1.sent[0]["right"]], rows[0, [0, 2]], atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(np.diff(goal["target_time"]), 1 / ctx.args.chunk_hz, atol=1e-8)
        count += 1
    if not count and not goto_count:
        raise ValueError("Empty capture")
    result = {"passed": True, "messages": count, "goto_messages": goto_count, "organizer_revision": revision,
              "physical_commands_sent": False, "physics_validated": False,
              "tracking_error_modelled": False, "timestamp_refreshed_for_offline_replay": True,
              "original_wire_envelope": args.capture is not None,
              "stats": ctx.stats.report()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
