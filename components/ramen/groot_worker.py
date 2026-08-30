"""self-contained GR00T pick worker client (直接 worker protocol、desktop 非依存)。

vendored worker script (`real_groot_n17_worker.py`) を worker venv で subprocess
起動し、`worker_protocol` で predict する。desktop の policy stack (base / config_loader /
Observation) には依存しない。返り値は **raw 38D** (robot_q_desired 36 + hand 2)。

依存:
- vendored `inference.desktop.upper_policy.{groot_pick_leg_contract,worker_protocol}`
  (compose_model_state / send_message / receive_message / 定数)。
- worker venv python = env `RAMEN_WORKER_PYTHON` (lerobot[groot] + torch + cv2 の Py3.12 env)。
  実 Thor container ではこの venv を image 内に構築し path を渡す (build スコープ)。
- huggingface_hub (checkpoint 取得) / cv2 (camera jpeg 化) / numpy。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_VENDOR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from inference.desktop.upper_policy.groot_pick_leg_contract import (  # noqa: E402
    CAMERA_ROLE_TO_KEY,
    MODEL_ACTION_DIM,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_STATE_DIM,
    TASK_TEXT,
    compose_model_state,
)
from inference.desktop.upper_policy.worker_protocol import (  # noqa: E402
    receive_message,
    send_message,
)

_WORKER_SCRIPT = (
    _VENDOR / "model/subtask_policy_training/deployment/real_groot_n17_worker.py"
)
# ver2-lora root layout の必須ファイル (checkpoint-40000 subdir 無し)。
_RUNTIME_FILES = (
    "config.json",
    "embodiment_id.json",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "processor_config.json",
    "statistics.json",
)


class GrootPickWorker:
    """GR00T pick worker を spawn し raw 38D action chunk を返す client。"""

    def __init__(
        self,
        worker_python: str | None = None,
        device: str = "cuda",
        repo_id: str = MODEL_REPO_ID,
        revision: str = MODEL_REVISION,
        task: str = TASK_TEXT,
    ):
        worker_python = worker_python or os.environ.get("RAMEN_WORKER_PYTHON")
        if not worker_python or not Path(worker_python).is_file():
            raise RuntimeError(
                "GR00T worker venv python not found. Set RAMEN_WORKER_PYTHON to the "
                "lerobot[groot] Py3.12 interpreter (dev: iros repo の "
                "model/subtask_policy_training/.venv/bin/python)."
            )
        if not _WORKER_SCRIPT.is_file():
            raise FileNotFoundError(f"vendored worker script missing: {_WORKER_SCRIPT}")

        self._task = task
        checkpoint = self._resolve_checkpoint(repo_id, revision)
        cmd = [
            str(worker_python), str(_WORKER_SCRIPT),
            "--checkpoint", str(checkpoint),
            "--device", device,
            "--model-repo-id", repo_id,
            "--model-revision", revision,
            "--task", task,
        ]
        self._p = subprocess.Popen(
            cmd, cwd=str(_VENDOR),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True,
        )
        self._rid = 0
        try:
            ready = receive_message(self._p.stdout)
            contract = (ready or {}).get("contract") or {}
            if ready.get("type") != "ready":
                raise RuntimeError(f"worker not ready: {ready!r}")
            if int(contract.get("state_dim", -1)) != MODEL_STATE_DIM:
                raise RuntimeError(f"worker state contract changed: {contract}")
            if int(contract.get("decoded_action_dim", -1)) != MODEL_ACTION_DIM:
                raise RuntimeError(f"worker action contract changed: {contract}")
            if int(contract.get("lower_body_command_dimensions", -1)) != 0:
                raise RuntimeError("worker may command the lower body")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _resolve_checkpoint(repo_id: str, revision: str) -> Path:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=_RUNTIME_FILES,
        ))
        required = (
            "config.json", "processor_config.json", "statistics.json",
            "embodiment_id.json", "model.safetensors.index.json",
        )
        missing = [n for n in required if not (snapshot / n).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete pick checkpoint {snapshot}: {missing}")
        return snapshot.resolve()

    def build_state(self, body_q29: np.ndarray, dex1_open_fraction) -> np.ndarray:
        """boundary body_q(29) + Dex1 開度 → model 38D state。"""
        return compose_model_state(
            np.asarray(body_q29, dtype=np.float32),
            np.clip(np.asarray(dex1_open_fraction, dtype=np.float32), 0.0, 1.0),
        )

    def predict(self, state38: np.ndarray, frames_bgr: dict) -> np.ndarray:
        """state(38) + frames_bgr{role:(480,640,3)uint8 BGR} → raw action (T, 38)。"""
        import cv2

        if self._p.stdin is None or self._p.stdout is None:
            raise RuntimeError("worker pipes are closed")
        self._rid += 1
        cameras = {}
        for role, key in CAMERA_ROLE_TO_KEY.items():
            img = np.asarray(frames_bgr[role], dtype=np.uint8)
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError(f"failed to jpeg-encode camera {role}")
            cameras[key] = enc.tobytes()
        send_message(self._p.stdin, {
            "type": "predict", "request_id": self._rid,
            "state": np.asarray(state38, dtype=np.float32),
            "cameras": cameras, "task": self._task,
        })
        resp = receive_message(self._p.stdout)
        if not isinstance(resp, dict) or resp.get("type") == "error":
            raise RuntimeError(f"worker predict failed: {resp}")
        if resp.get("type") != "prediction" \
                or int(resp.get("request_id", -1)) != self._rid:
            raise RuntimeError(f"unexpected worker response: {resp!r}")
        actions = np.asarray(resp.get("actions"), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != MODEL_ACTION_DIM \
                or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid worker action shape/value: {actions.shape}")
        return actions

    def close(self) -> None:
        p = getattr(self, "_p", None)
        if p is None or p.poll() is not None:
            return
        try:
            if p.stdin is not None and p.stdout is not None:
                send_message(p.stdin, {"type": "close"})
                receive_message(p.stdout)
        except Exception:
            pass
        try:
            p.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                p.kill()
