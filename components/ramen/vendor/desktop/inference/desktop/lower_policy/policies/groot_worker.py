"""Run GR00T in its Python 3.12 Pixi environment for the Python 3.10 runtime."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import socket
import traceback
from pathlib import Path

import numpy as np

from inference.desktop.lower_policy.policies.base import CameraKey, Observation, PolicyConfig
from inference.desktop.lower_policy.policies.groot import CAMERAS, STATE_DIM, Gr00tPolicy
from inference.desktop.lower_policy.policies.groot_worker_protocol import (
    receive_archive,
    scalar_text,
    send_archive,
)


def _terminate_with_parent() -> None:
    """Do not leave a 6-GiB GPU worker behind if the runtime is killed."""

    try:
        libc = ctypes.CDLL(None)
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
    except (AttributeError, OSError):
        pass


def _validate_image(request: dict[str, np.ndarray], key: str) -> np.ndarray:
    image = np.asarray(request[key])
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError(
            f"{key} must be uint8 (480, 640, 3), got {image.shape} {image.dtype}"
        )
    return np.ascontiguousarray(image)


def serve(policy: Gr00tPolicy, socket_path: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(1)
        print(f"[groot-worker] ready socket={socket_path}", flush=True)
        try:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(600.0)
                while True:
                    request = receive_archive(connection)
                    if request is None:
                        break
                    try:
                        kind = scalar_text(request.get("kind", np.asarray("predict")), "kind")
                        if kind == "warmup":
                            count = int(np.asarray(request.get("count", [1])).reshape(-1)[0])
                            policy.warmup(n_iter=count)
                            send_archive(connection, ok=np.asarray([1], dtype=np.uint8))
                        elif kind == "predict":
                            state = np.asarray(request["state"], dtype=np.float32)
                            if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
                                raise ValueError(f"state must be finite ({STATE_DIM},), got {state.shape}")
                            frames = {
                                CameraKey.HEAD_LEFT: _validate_image(request, "head_left"),
                                CameraKey.WRIST_LEFT: _validate_image(request, "wrist_left"),
                                CameraKey.WRIST_RIGHT: _validate_image(request, "wrist_right"),
                            }
                            frames_prev = None
                            if policy._video_horizon == 2:
                                frames_prev = {
                                    CameraKey.HEAD_LEFT: _validate_image(request, "head_left_prev"),
                                    CameraKey.WRIST_LEFT: _validate_image(request, "wrist_left_prev"),
                                    CameraKey.WRIST_RIGHT: _validate_image(request, "wrist_right_prev"),
                                }
                            observation = Observation(
                                frames_bgr=frames,
                                frames_bgr_prev=frames_prev,
                                state=state,
                                skill_id=None,
                                language=scalar_text(request["language"], "language"),
                                obb_detections=None,
                                timestamp_ns=int(np.asarray(request["timestamp_ns"]).reshape(-1)[0]),
                            )
                            # Return the complete decoded chunk.  Temporal
                            # ensembling and async replanning belong to the
                            # parent DDS runtime so there is exactly one state
                            # machine and one control-step clock.
                            chunk, latency_ms, raw_chunk = (
                                policy._sync_predict_chunk_19d(observation)
                            )
                            send_archive(
                                connection,
                                ok=np.asarray([1], dtype=np.uint8),
                                action_chunk=chunk,
                                latency_ms=np.asarray([latency_ms], dtype=np.float64),
                                metadata_json=np.asarray(json.dumps({
                                    "raw_action_shape_53d": list(raw_chunk.shape),
                                })),
                            )
                        elif kind == "close":
                            send_archive(connection, ok=np.asarray([1], dtype=np.uint8))
                            break
                        else:
                            raise ValueError(f"unsupported GR00T worker request: {kind!r}")
                    except Exception as exc:
                        traceback.print_exc()
                        send_archive(
                            connection,
                            ok=np.asarray([0], dtype=np.uint8),
                            error=np.asarray(str(exc)),
                        )
        finally:
            socket_path.unlink(missing_ok=True)
            policy.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--ckpt-ref", required=True)
    parser.add_argument("--checkpoint-subdir")
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    args = parser.parse_args()
    _terminate_with_parent()
    cfg = PolicyConfig(
        mode=args.mode,
        ckpt_ref=args.ckpt_ref,
        checkpoint_subdir=args.checkpoint_subdir,
        device=args.device,
        dtype=args.dtype,
        cams=CAMERAS,
    )
    serve(Gr00tPolicy.from_ckpt(cfg), args.socket)


if __name__ == "__main__":
    main()
