"""self-contained 53D LeRobot GR00T worker client (原さん inference server 再利用)。

⚠️ WIP / BLOCKED (2026-08-31、未使用): 現行 Groot53Backend は desktop 版を使う。
本 self-contained 経路は2 blocker で保留:
  (1) lerobot 版分裂: 53D checkpoint=0.6.1、pick=0.6.0 → container で env 分裂。
  (2) checkpoint load 経路: takada 53D は embed_tokens tied-weight を持ち、原さん server の
      vanilla GrootPolicy.from_pretrained が strict load 失敗。desktop groot.py の custom
      load (raw_config + streaming shards、strict 回避) が必須。本 server にその load を
      port するのが follow-up。詳細は VENDOR_NOTES.md。


vendored `groot53/groot53_server.py`(原さん eval server、lerobot 0.6.0、self-contained)を
Unix socket で spawn し、predict する。desktop policy stack 非依存。

parent 側:
- checkpoint 解決 (snapshot_download、LeRobot GR00T 形式、subdir 対応)。
- obs body_q(29) → G1WristFK で ee_state(12) → build_state_49d で 49D state。
- server に {state, 3 cam(BGR→? 実際は uint8 RGB), task} を送り、decoded 53D を受信。
- slice_53d_to_19d で 19D executable (waist3+arm14+hand2、絶対)。

worker python は env `RAMEN_WORKER_PYTHON`(lerobot[groot] を持つ Py3.12、Thor container は python3)。

53D skill の ckpt 対応表 (self-contained)。ckpt_ref は inference/desktop policy_config.yaml と一致。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from .g1_urdf_fk import G1WristFK

_VENDOR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from groot53.groot53_server import receive_archive, send_archive  # noqa: E402
from groot53.state53 import build_state_49d, slice_53d_to_19d  # noqa: E402

_SERVER = _VENDOR / "groot53" / "groot53_server.py"

# skill variant → (HF ckpt_ref, checkpoint_subdir or None, default task)。
# inference/desktop policy_config.yaml と一致 (self-contained registry)。
VARIANT_REGISTRY: dict[str, tuple[str, str | None, str]] = {
    "groot_rotate_leg_200k": (
        "Team-RAMEN/IROS2026_RAMEN_takada_rotate_leg_to_tighten_optimal_gr00t_200k"
        "@51e306fd19dfbcba81932c6b6f073615fdb54b26",
        None, "rotate the table leg to tighten",
    ),
    "groot_insert_leg_200k": (
        "Team-RAMEN/IROS2026_RAMEN_takada_insert_leg_optimal_gr00t_200k"
        "@0f0927bcc1aed6927fb26b82c017009961ee3d35",
        None, "insert the table leg",
    ),
    "groot_overlay": (
        "Team-RAMEN/IROS2026_RAMEN_hara_task_5_7_groot_overlay_v11b_100k_v1",
        None, "rotate and move the table base",
    ),
    "groot_flip_table_n17_2_baseline": (
        "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_groot_n17_2_baseline_checkpoints"
        "@1a408d87eda8d01f9b79113f1aed97a5d0811bff",
        "checkpoints/020000/pretrained_model", "flip the table",
    ),
}

# LeRobot GR00T 形式の必須ファイル。
_RUNTIME_FILES = (
    "config.json", "model.safetensors", "model-*.safetensors",
    "model.safetensors.index.json", "policy_preprocessor.json",
    "policy_postprocessor.json", "*.json",
)


class Groot53Worker:
    """53D LeRobot GR00T worker (socket)。predict → 19D executable。"""

    def __init__(
        self,
        variant: str,
        worker_python: str | None = None,
        device: str = "cuda:0",
        n_action_steps: int = 16,
        task: str | None = None,
    ):
        import shutil

        if variant not in VARIANT_REGISTRY:
            raise ValueError(
                f"unknown 53D variant {variant!r}; known: {sorted(VARIANT_REGISTRY)}"
            )
        ckpt_ref, subdir, default_task = VARIANT_REGISTRY[variant]
        self._task = task or default_task

        worker_python = worker_python or os.environ.get("RAMEN_WORKER_PYTHON")
        if worker_python and not Path(worker_python).is_file():
            worker_python = shutil.which(worker_python) or worker_python
        if not worker_python or not Path(worker_python).is_file():
            raise RuntimeError(
                "GR00T worker python not found. Set RAMEN_WORKER_PYTHON "
                "(dev: iros .venv; Thor container: python3)."
            )
        if not _SERVER.is_file():
            raise FileNotFoundError(f"vendored 53D server missing: {_SERVER}")

        self._fk = G1WristFK.from_urdf()
        checkpoint = self._resolve_checkpoint(ckpt_ref, subdir)
        self._socket_path = Path(
            f"/tmp/ramen_groot53_{os.getpid()}_{uuid.uuid4().hex}.sock"
        )
        cmd = [
            str(worker_python), str(_SERVER),
            "--checkpoint", str(checkpoint),
            "--socket", str(self._socket_path),
            "--device", device,
            "--n-action-steps", str(n_action_steps),
        ]
        self._proc = subprocess.Popen(
            cmd, cwd=str(_VENDOR),
            stdout=subprocess.PIPE, stderr=None, start_new_session=True,
        )
        self._wait_ready(timeout=600.0)
        self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._conn.connect(str(self._socket_path))

    @staticmethod
    def _resolve_checkpoint(ckpt_ref: str, subdir: str | None) -> Path:
        from huggingface_hub import snapshot_download

        repo_id, revision = ckpt_ref, None
        if "@" in ckpt_ref:
            repo_id, revision = ckpt_ref.rsplit("@", 1)
        patterns = tuple(
            (f"{subdir}/{p}" if subdir else p) for p in _RUNTIME_FILES
        )
        snapshot = Path(snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=patterns,
        ))
        root = snapshot / subdir if subdir else snapshot
        if not (root / "config.json").is_file():
            raise FileNotFoundError(f"53D checkpoint incomplete at {root}")
        return root.resolve()

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        assert self._proc.stdout is not None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"53D server exited during startup (code={self._proc.returncode})"
                )
            line = self._proc.stdout.readline().decode("utf-8", "replace")
            if "inference server ready" in line:
                return
        raise TimeoutError("timed out waiting for 53D server ready")

    def build_state(self, body_q29: np.ndarray, dex1_open_fraction) -> np.ndarray:
        ee_state = self._fk.compute_ee_state(np.asarray(body_q29, np.float32))
        return build_state_49d(body_q29, dex1_open_fraction, ee_state)

    def predict(self, state49: np.ndarray, frames_rgb: dict) -> np.ndarray:
        """state(49) + frames_rgb{head_left/left_wrist/right_wrist:(480,640,3)uint8 RGB}
        → 19D executable (T,19)。"""
        send_archive(
            self._conn,
            kind=np.asarray("predict"),
            state=np.asarray(state49, dtype=np.float32),
            task=np.asarray(self._task),
            head_left=np.ascontiguousarray(frames_rgb["head_left"], dtype=np.uint8),
            left_wrist=np.ascontiguousarray(frames_rgb["left_wrist"], dtype=np.uint8),
            right_wrist=np.ascontiguousarray(frames_rgb["right_wrist"], dtype=np.uint8),
        )
        reply = receive_archive(self._conn)
        if not reply or "action" not in reply:
            raise RuntimeError(f"53D server predict failed: {reply}")
        decoded53 = np.asarray(reply["action"], dtype=np.float64)   # (T,53)
        if decoded53.ndim != 2 or decoded53.shape[1] != 53:
            raise RuntimeError(f"expected (T,53) decoded, got {decoded53.shape}")
        return slice_53d_to_19d(decoded53)

    def reset(self) -> None:
        try:
            send_archive(self._conn, kind=np.asarray("reset"))
            receive_archive(self._conn)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        p = getattr(self, "_proc", None)
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                p.kill()
        try:
            self._socket_path.unlink(missing_ok=True)
        except Exception:
            pass
