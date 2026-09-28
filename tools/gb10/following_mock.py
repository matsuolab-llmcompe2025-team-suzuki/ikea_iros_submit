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
from wire_probe import decode_joint, decode_goto


class ScheduledFollower:
    """Lagged software follower, with the same *assumed* sag model as the sender.

    This is not a PhysX/WBC simulation and cannot validate real tracking. The
    pinned organizer adapter is exercised separately against fake robot I/O.
    """

    def __init__(self, gravity=None):
        self.q = np.zeros(14)
        self.target = self.q.copy()
        self.start = self.q.copy()
        self.started_at = 0.0
        self.duration = 0.0
        self.gravity = gravity

    def goto(self, target, speed, now):
        self.start = self.q.copy()
        self.target = np.asarray(target).copy()
        self.started_at = now
        self.duration = max(1, int(np.ceil(np.max(np.abs(self.target - self.start)) / speed * 20))) / 20
        if self.duration > 15:
            raise ValueError("goto exceeds official 15 second limit")

    def step(self, now, dt):
        fraction = min(1.0, (now - self.started_at) / self.duration) if self.duration else 1.0
        desired = self.start + fraction * (self.target - self.start)
        if self.gravity is not None:
            _, offset = self.gravity.apply(self.q)
            desired = desired - offset
        self.q += np.clip((desired - self.q) * min(1.0, dt / 0.12), -0.8 * dt, 0.8 * dt)
        return self.q


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
    sys.path.insert(0, "/app/ramen")
    import yaml
    from pathlib import Path
    from inference.desktop.lower_policy.actuators.boundary_sink import ArmGravitySagOffset

    config = yaml.safe_load(Path("/app/ramen/inference/desktop/lower_policy/configs/skill_config.yaml").read_text())
    follower = ScheduledFollower(ArmGravitySagOffset.from_config(config))
    hands = np.full(2, 4.5)
    hand_target = hands.copy()
    start = last_state = last_camera = time.monotonic()
    messages = 0
    try:
        while not stop:
            now = time.monotonic()
            if actions.poll(1):
                message = actions.recv()
                if message.startswith(b"goto"):
                    target, speed, _ = decode_goto(message)
                    follower.goto(target, speed, now)
                    messages += 1
                    continue
                chunk, _ = decode_joint(message)
                # The boundary publishes repeated rows. Refuse a trajectory here:
                # this follower does not implement the organizer's chunk scheduler.
                if not np.allclose(chunk, chunk[0], atol=0, rtol=0):
                    raise ValueError("Follower only supports repeated hold rows")
                follower.target = chunk[0, 4:18].copy()
                follower.duration = 0.0
                hand_target = (1 - chunk[0, [0, 2]]) * 2.65
                messages += 1
            if now - last_state >= 0.02:
                dt = now - last_state
                q[15:29] = follower.step(now, dt)
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
