#!/usr/bin/env python3
"""Verify that both official Dex1-1 DDS state topics are live.

This deliberately creates subscribers only: it never creates a publisher and
therefore cannot command either gripper.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read both Dex1-1 states without sending any command."
    )
    parser.add_argument("--interface", required=True, help="CycloneDDS NIC name")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.interface)
    topics = {
        "left": "rt/dex1/left/state",
        "right": "rt/dex1/right/state",
    }
    received: queue.Queue[tuple[str, MotorStates_ | None]] = queue.Queue()

    # CycloneDDS lazily builds XTypes metadata the first time an IDL class is
    # used. Constructing two subscribers for the same MotorStates_ class in
    # parallel can race inside that one-time initialization and fail with
    # ``version_support is None``. Subscriber creation itself does not block,
    # so serialize it here and only perform the potentially blocking Read calls
    # in daemon threads.
    subscribers: dict[str, ChannelSubscriber] = {}
    for side, topic in topics.items():
        subscriber = ChannelSubscriber(topic, MotorStates_)
        subscriber.Init()
        subscribers[side] = subscriber

    def reader(side: str, subscriber: ChannelSubscriber) -> None:
        received.put((side, subscriber.Read()))

    # ChannelSubscriber.Read may block forever before the PC2 service is up.
    # Daemon threads let this diagnostic fail cleanly without a publisher or
    # any command path to the robot.
    for side, subscriber in subscribers.items():
        threading.Thread(
            target=reader, args=(side, subscriber), daemon=True
        ).start()

    states: dict[str, MotorStates_ | None] = {}
    deadline = time.monotonic() + args.timeout
    while len(states) < len(topics) and time.monotonic() < deadline:
        try:
            side, message = received.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            break
        states[side] = message
    if len(states) < len(topics):
        missing = [side for side in topics if side not in states]
        print(
            "Dex1 state timeout: no sample from " + ", ".join(missing) + ". "
            "Start dex1_1_gripper_server on Orin/PC2.",
            file=sys.stderr,
        )
        return 1

    if any(message is None or not message.states for message in states.values()):
        print("Dex1 state topic returned an empty sample.", file=sys.stderr)
        return 1

    print(
        "dex1-state-ok "
        f"left_q={states['left'].states[0].q:.6f} "
        f"right_q={states['right'].states[0].q:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
