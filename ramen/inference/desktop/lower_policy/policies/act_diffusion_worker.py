"""ACT / Diffusion Policy を Python 3.12 の推論 env で動かす worker (Issue #139)。

Python 3.10 の DDS runtime (`ActDiffusionPolicy`) から Unix socket で呼ばれる。
要求は describe / predict / close の 3 種。overlay 描画・観測履歴・async 再計画は
runtime 側の責務で、ここには model 入力そのものの画像が届く。
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import socket
import traceback
from pathlib import Path

import numpy as np

from inference.desktop.lower_policy.policies.act_diffusion import (
    ACTION_DIM,
    CAMERAS,
    IMAGE_SHAPE_HWC,
    STATE_DIM,
    ActDiffusionModel,
)
from inference.desktop.lower_policy.policies.groot_worker_protocol import (
    receive_archive,
    scalar_text,
    send_archive,
)


def _terminate_with_parent() -> None:
    """runtime が落ちた時に GPU を掴んだ worker を残さない。"""
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
    except (AttributeError, OSError):
        pass


def _predict(model: ActDiffusionModel, request: dict[str, np.ndarray]) -> np.ndarray:
    states = np.asarray(request["states"], dtype=np.float32)
    if states.shape != (model.n_obs_steps, STATE_DIM) or not np.isfinite(states).all():
        raise ValueError(
            f"states must be finite ({model.n_obs_steps}, {STATE_DIM}), got {states.shape}"
        )
    frames_seq = []
    for index in range(model.n_obs_steps):
        frames = {}
        for cam in CAMERAS:
            image = np.asarray(request[f"{cam.value}_{index}"])
            if image.shape != IMAGE_SHAPE_HWC or image.dtype != np.uint8:
                raise ValueError(
                    f"{cam.value}_{index} must be uint8 {IMAGE_SHAPE_HWC}, "
                    f"got {image.shape} {image.dtype}"
                )
            frames[cam] = image
        frames_seq.append(frames)
    chunk, _latency_ms = model.predict_chunk(frames_seq, list(states))
    if chunk.shape != (model.chunk_len, ACTION_DIM) or not np.isfinite(chunk).all():
        raise RuntimeError(f"model returned an invalid chunk: shape={chunk.shape}")
    return chunk


def serve(model: ActDiffusionModel, socket_path: Path, accept_timeout_s: float = 300.0) -> None:
    """runtime 1 本からの要求を、接続が切れるまで処理する。

    次の要求は timeout 無しで待つ (operator が run_skill の gate で何分待っても落ちない)。
    runtime が落ちれば接続が切れて EOF になるので、それで終わる。接続前に runtime が
    落ちた場合に備えて、accept だけ `accept_timeout_s` で打ち切る。
    """
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(1)
        server.settimeout(accept_timeout_s)
        print(f"[act-diffusion-worker] ready socket={socket_path}", flush=True)
        try:
            try:
                connection, _ = server.accept()
            except TimeoutError:
                print(
                    f"[act-diffusion-worker] no runtime connected in {accept_timeout_s:.0f}s; exiting",
                    flush=True,
                )
                return
            with connection:
                connection.settimeout(None)
                while True:
                    request = receive_archive(connection)
                    if request is None:
                        break
                    try:
                        kind = scalar_text(request["kind"], "kind")
                        if kind == "describe":
                            send_archive(
                                connection,
                                ok=np.asarray([1], dtype=np.uint8),
                                policy_kind=np.asarray(model.policy_kind),
                                n_obs_steps=np.asarray([model.n_obs_steps], dtype=np.int64),
                                chunk_len=np.asarray([model.chunk_len], dtype=np.int64),
                            )
                        elif kind == "predict":
                            send_archive(
                                connection,
                                ok=np.asarray([1], dtype=np.uint8),
                                action_chunk=_predict(model, request),
                            )
                        elif kind == "close":
                            send_archive(connection, ok=np.asarray([1], dtype=np.uint8))
                            break
                        else:
                            raise ValueError(f"unsupported worker request: {kind!r}")
                    except Exception as exc:
                        traceback.print_exc()
                        send_archive(
                            connection,
                            ok=np.asarray([0], dtype=np.uint8),
                            error=np.asarray(str(exc)),
                        )
        finally:
            socket_path.unlink(missing_ok=True)
            model.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ckpt-ref", required=True)
    parser.add_argument("--checkpoint-subdir")
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    _terminate_with_parent()
    model = ActDiffusionModel.from_checkpoint(
        args.ckpt_ref, args.checkpoint_subdir, args.device
    )
    serve(model, args.socket)


if __name__ == "__main__":
    main()
