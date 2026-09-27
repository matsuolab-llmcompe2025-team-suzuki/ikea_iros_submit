#!/usr/bin/env python3
"""Loopback-only joint follower for software tests, NOT physics or robot dynamics.

SIGUSR1 toggles camera publishing; SIGUSR2 toggles state publishing for fault tests.
All sockets stay on 127.0.0.1. No SDK, DDS or robot connection is created.
"""

import json
import signal
import sys
import time

import cv2
import msgpack
import numpy as np
import zmq

sys.path.insert(0, "/app")
from mocks.mock_orin import synthetic_frame
from wire_probe import decode_joint


def main():
    stop = False
    camera_enabled = True
    state_enabled = True

    def handle(sig, _frame):
        nonlocal stop, camera_enabled, state_enabled
        if sig == signal.SIGUSR1:
            camera_enabled = not camera_enabled
        elif sig == signal.SIGUSR2:
            state_enabled = not state_enabled
        else:
            stop = True
        print(json.dumps({"signal": sig, "camera_enabled": camera_enabled,
                          "state_enabled": state_enabled}), flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2):
        signal.signal(sig, handle)
    context = zmq.Context()
    cameras = context.socket(zmq.PUB)
    states = context.socket(zmq.PUB)
    actions = context.socket(zmq.SUB)
    for socket in (cameras, states, actions):
        socket.setsockopt(zmq.LINGER, 0)
    actions.setsockopt(zmq.SUBSCRIBE, b"")
    cameras.bind("tcp://127.0.0.1:5555")
    states.bind("tcp://127.0.0.1:5557")
    actions.connect("tcp://127.0.0.1:5556")
    keys = ("ego_view", "ego_view_left", "ego_view_right", "left_wrist", "right_wrist")
    q = np.zeros(29)
    target = q[15:29].copy()
    hands = np.full(2, 4.5)
    hand_target = hands.copy()
    start = last_state = last_camera = time.monotonic()
    messages = 0
    try:
        while not stop:
            now = time.monotonic()
            if actions.poll(1):
                chunk, _ = decode_joint(actions.recv())
                # The boundary publishes repeated rows. Refuse a trajectory here:
                # this follower does not implement the organizer's chunk scheduler.
                if not np.allclose(chunk, chunk[0], atol=0, rtol=0):
                    raise ValueError("Follower only supports repeated hold rows")
                target = chunk[0, 4:18].copy()
                hand_target = (1 - chunk[0, [0, 2]]) * 2.65
                messages += 1
            if now - last_state >= 0.02:
                dt = now - last_state
                q[15:29] += np.clip(target - q[15:29], -0.8 * dt, 0.8 * dt)
                hands += np.clip(hand_target - hands, -3 * dt, 3 * dt)
                if state_enabled:
                    payload = {"body_q": q.tolist(), "base_quat": [1., 0., 0., 0.],
                               "gripper_q": {side: {"q": -float(value)}
                                             for side, value in zip(("left", "right"), hands)}}
                    states.send(b"g1_debug" + msgpack.packb(payload, use_bin_type=True))
                last_state = now
            if now - last_camera >= 1 / 30:
                if camera_enabled:
                    images = {}
                    for key in keys:
                        ok, jpeg = cv2.imencode(".jpg", synthetic_frame(key, now - start),
                                               [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if not ok:
                            raise RuntimeError("JPEG encoding failed")
                        images[key] = jpeg.tobytes()
                    cameras.send(msgpack.packb({"images": images,
                                                "timestamps": {k: time.time() for k in keys}},
                                               use_bin_type=True))
                last_camera = now
    finally:
        for socket in (cameras, states, actions):
            socket.close()
        context.term()
        print(json.dumps({"messages": messages, "physical_commands_sent": False}), flush=True)


if __name__ == "__main__":
    main()
