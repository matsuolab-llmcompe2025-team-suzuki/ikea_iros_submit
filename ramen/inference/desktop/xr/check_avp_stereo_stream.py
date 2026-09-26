#!/usr/bin/env python3
"""Read-only health check for the Apple Vision Pro head-stereo stream.

This intentionally imports no Unitree SDK and sends no robot command.  The
official image service sends one side-by-side 1280x480 frame; a valid
binocular configuration must split it into two 640x480 views.
"""

from __future__ import annotations

import argparse
import time

from teleimager.image_client import ImageClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Orin TeleImager IP address")
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    client = ImageClient(host=args.host)
    config = client.get_cam_config()["head_camera"]
    if config.get("binocular") is not True:
        raise SystemExit(f"head camera is not binocular: {config!r}")
    if tuple(config.get("image_shape", ())) != (480, 1280):
        raise SystemExit(
            "expected a 1280x480 side-by-side head image; "
            f"got image_shape={config.get('image_shape')!r}"
        )

    deadline = time.monotonic() + args.timeout
    frame = None
    source_fps = 0.0
    while time.monotonic() < deadline:
        frame, source_fps = client.get_head_frame()
        if frame is not None:
            break
        time.sleep(0.02)

    if frame is None:
        raise SystemExit(f"no head-stereo frame received within {args.timeout:.1f}s")
    if frame.shape != (480, 1280, 3):
        raise SystemExit(f"unexpected head frame shape: {frame.shape!r}")

    left = frame[:, :640]
    right = frame[:, 640:]
    if left.shape != (480, 640, 3) or right.shape != (480, 640, 3):
        raise SystemExit(f"invalid stereo split: left={left.shape}, right={right.shape}")

    print(
        "OK: AVP head stereo stream "
        f"left={left.shape} right={right.shape} source_fps={source_fps:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
