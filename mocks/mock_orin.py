#!/usr/bin/env python3
"""Stand-in for the organizer's Orin — cameras on :5555 and state on :5557.

One process, both input endpoints, no robot and no hardware. Run this and
your client cannot tell it is not on the G1 (until it tries to move).

    python mocks/mock_orin.py                 # Dex1-1 rig: no hand state
    python mocks/mock_orin.py --with-hands    # Dex3 rig: hand state present
    python mocks/mock_orin.py --no-wrists     # ego camera only
    python mocks/mock_orin.py --stereo-ego    # also publish ego_view_left/right

The images are a moving synthetic pattern rather than noise, so you can see
at a glance whether your client is decoding frames or holding a stale one.
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

WIDTH, HEIGHT = 640, 480
BODY_DOF = 29
HAND_DOF = 7
STATE_TOPIC = b"g1_debug"


def synthetic_frame(key: str, phase: float) -> np.ndarray:
    """A labelled gradient with a moving bar — obviously fake, obviously live."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, WIDTH, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.linspace(0, 255, HEIGHT, dtype=np.uint8)[:, None]
    x = int((0.5 + 0.5 * np.sin(phase)) * (WIDTH - 40))
    frame[:, x:x + 40, 2] = 255
    cv2.putText(frame, key, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"t={phase:7.2f}", (20, HEIGHT - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def publish_cameras(port: int, fps: int, keys: list[str], stop: threading.Event):
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 20)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{port}")
    print(f"[mock-orin] cameras PUB tcp://*:{port} keys={keys} at {fps} Hz")

    period, start, sent = 1.0 / fps, time.monotonic(), 0
    while not stop.is_set():
        tick = time.monotonic()
        phase = tick - start
        images, timestamps = {}, {}
        for key in keys:
            # BGR on the wire, matching the real server; the client flips it.
            ok, jpeg = cv2.imencode(
                ".jpg", synthetic_frame(key, phase),
                [int(cv2.IMWRITE_JPEG_QUALITY), 80],
            )
            if ok:
                images[key] = jpeg.tobytes()
                timestamps[key] = time.time()
        socket.send(msgpack.packb({"timestamps": timestamps, "images": images},
                                  use_bin_type=True))
        sent += 1
        if sent % 150 == 0:
            print(f"[mock-orin] {sent} camera frames")
        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)
    socket.close(linger=0)


def publish_state(port: int, rate_hz: float, with_hands: bool, stop: threading.Event):
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 20)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{port}")
    print(f"[mock-orin] state PUB tcp://*:{port} at {rate_hz} Hz "
          f"(hands {'present' if with_hands else 'absent — Dex1-1 rig'})")

    period, start, sent = 1.0 / rate_hz, time.monotonic(), 0
    while not stop.is_set():
        tick = time.monotonic()
        phase = tick - start
        payload = {
            "body_q": (0.15 * np.sin(phase + np.arange(BODY_DOF) * 0.2)).tolist(),
            "base_quat": [1.0, 0.0, 0.0, 0.0],
        }
        if with_hands:
            payload["left_hand_q"] = [0.5] * HAND_DOF
            payload["right_hand_q"] = [0.5] * HAND_DOF
        socket.send(STATE_TOPIC + msgpack.packb(payload, use_bin_type=True))
        sent += 1
        if sent % 250 == 0:
            print(f"[mock-orin] {sent} states")
        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)
    socket.close(linger=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--state-port", type=int, default=5557)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--state-hz", type=float, default=50.0)
    parser.add_argument("--with-hands", action="store_true",
                        help="Publish Dex3 hand state (the real rig has Dex1-1 and does not).")
    parser.add_argument("--no-wrists", action="store_true",
                        help="Publish ego_view only.")
    parser.add_argument("--stereo-ego", action="store_true",
                        help="Also publish ego_view_left/ego_view_right, "
                             "for testing a stereo camera_keys declaration.")
    args = parser.parse_args()

    keys = ["ego_view"] if args.no_wrists else ["ego_view", "left_wrist", "right_wrist"]
    if args.stereo_ego:
        keys += ["ego_view_left", "ego_view_right"]
    stop = threading.Event()
    threads = [
        threading.Thread(target=publish_cameras,
                         args=(args.camera_port, args.fps, keys, stop), daemon=True),
        threading.Thread(target=publish_state,
                         args=(args.state_port, args.state_hz, args.with_hands, stop),
                         daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop.set()
        for thread in threads:
            thread.join(timeout=2.0)
        print("\n[mock-orin] stopped")


if __name__ == "__main__":
    main()
