#!/usr/bin/env python3
"""Serve Orin's binocular head camera to Apple Vision Pro without G1 control.

This program does not import or initialize Unitree SDK code.  It serves the
official TeleVuer client endpoint on the Desktop and uses the Orin WebRTC
head-camera stream, matching the official XR teleoperation transport.  Stop
it with Ctrl-C after confirming the Vision Pro stereo view.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import time

from teleimager.image_client import ImageClient
from televuer import TeleVuerWrapper


def sync_vuer_index_gzip() -> None:
    """Regenerate Vuer's pre-compressed entry page from its live source.

    Vision Pro Safari requests the gzip variant when it is present.  Vuer does
    not generate that variant itself, so a stale ``index.html.gz`` can make a
    browser run an older client than the server.  Rebuild it atomically before
    the viewer starts; doing this at each service start also makes package
    upgrades safe.
    """
    spec = importlib.util.find_spec("vuer")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not locate the installed vuer package")
    client_root = Path(spec.origin).resolve().parent / "client_build"
    source = client_root / "index.html"
    target = client_root / "index.html.gz"
    if not source.is_file():
        raise RuntimeError(f"Vuer client entry page is missing: {source}")

    with source.open("rb") as input_file:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=client_root, prefix=".index.html.", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with gzip.GzipFile(
                fileobj=temporary_file, mode="wb", compresslevel=9, mtime=0
            ) as compressed_file:
                shutil.copyfileobj(input_file, compressed_file)
    os.replace(temporary_path, target)
    print(f"Synchronized compressed Vuer entry page: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-server-ip", default="192.168.29.159")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sync_vuer_index_gzip()
    desktop_ip = os.environ.get("AVP_DESKTOP_IP", "192.168.29.175")
    client = ImageClient(host=args.image_server_ip)
    head = client.get_cam_config()["head_camera"]
    if head.get("binocular") is not True or tuple(head.get("image_shape", ())) != (480, 1280):
        raise SystemExit(f"expected 1280x480 binocular head stream, got {head!r}")
    if not head.get("enable_webrtc"):
        raise SystemExit("head-camera WebRTC is disabled on the Orin image service")

    viewer = TeleVuerWrapper(
        use_hand_tracking=True,
        binocular=True,
        img_shape=(480, 1280),
        display_mode="immersive",
        zmq=False,
        webrtc=True,
        webrtc_url=f"https://{args.image_server_ip}:{head['webrtc_port']}/offer",
    )
    print("AVP view-only server ready (no G1 SDK and no robot command).")
    print(
        "Open this exact URL on Apple Vision Pro:\n"
        f"https://{desktop_ip}:8012/?ws=wss://{desktop_ip}:8012"
    )
    print("First trust the Orin WebRTC page, then enter Virtual Reality.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping view-only server.")
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
