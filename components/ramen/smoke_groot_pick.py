"""実 GR00T pick 推論 → (T,25) smoke (GPU 必要)。

worker venv (lerobot[groot]) を持つ環境から実行する想定:
    cd /datadrive2/iros_2026_ramen && \
    RAMEN_WORKER_PYTHON=/datadrive2/iros_2026_ramen/model/subtask_policy_training/.venv/bin/python \
    pixi run -e runtime python /datadrive2/ikea_iros_submit/components/ramen/smoke_groot_pick.py

mock obs (zeros 画像 + 妥当な body_q) で GrootWorkerBackend を1回 forward し、
boundary DecoupledSink.validate_chunk を通ることを確認する (task 成否は問わない)。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SUBMIT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SUBMIT_ROOT not in sys.path:
    sys.path.insert(0, _SUBMIT_ROOT)

from components.ramen.policy import GrootPickTaskspacePolicy, GrootWorkerBackend

# boundary/actions.py (msgpack 依存で runtime env 非対応) を import せず、
# DecoupledSink.validate_chunk と同一の契約を inline 検証する。
_TASKSPACE_SLICES = {
    "left_hand": slice(0, 2), "right_hand": slice(2, 4),
    "left_ee_pos": slice(4, 7), "left_ee_quat": slice(7, 11),
    "right_ee_pos": slice(11, 14), "right_ee_quat": slice(14, 18),
}


def _validate_chunk(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 2 and arr.shape[1] == 25, f"bad shape {arr.shape}"
    assert 1 <= arr.shape[0] <= 64, f"bad T {arr.shape[0]}"
    assert np.issubdtype(arr.dtype, np.floating) and np.all(np.isfinite(arr))
    hands = np.concatenate([arr[:, 0:2], arr[:, 2:4]], axis=1)
    assert np.max(np.abs(hands)) <= 1.0 + 1e-3, "hands out of [-1,1]"
    for sl in (slice(7, 11), slice(14, 18)):
        norms = np.linalg.norm(arr[:, sl], axis=1)
        assert np.all(np.abs(norms - 1.0) <= 1e-2), "EE quat not unit"
    return arr


def main() -> int:
    TASKSPACE_SLICES = _TASKSPACE_SLICES
    print("[smoke] building GrootWorkerBackend (loads real pick worker) ...")
    t0 = time.monotonic()
    backend = GrootWorkerBackend()
    policy = GrootPickTaskspacePolicy(lane="decoupled", backend=backend)
    print(f"[smoke] backend ready in {time.monotonic() - t0:.1f}s")

    obs = {
        "images": {
            "ego_view": np.zeros((480, 640, 3), dtype=np.uint8),
            "left_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            "right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        },
        "body_q": np.zeros(29, dtype=np.float32),
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "prompt": "pick table leg",
        "t": 0.0,
    }

    t1 = time.monotonic()
    actions = policy.act(obs)["actions"]
    dt = (time.monotonic() - t1) * 1e3
    print(f"[smoke] act() -> actions {actions.shape} {actions.dtype} in {dt:.0f}ms")

    validated = _validate_chunk(actions)   # raises on violation
    print("[smoke] (T,25) boundary contract PASSED (inline)")

    ls, rs = TASKSPACE_SLICES["left_ee_pos"], TASKSPACE_SLICES["right_ee_pos"]
    print(f"[smoke] left_ee_pos[0]  = {validated[0, ls]}")
    print(f"[smoke] right_ee_pos[0] = {validated[0, rs]}")
    print(
        "[smoke] left_ee_quat|norm| =",
        float(np.linalg.norm(validated[0, TASKSPACE_SLICES['left_ee_quat']])),
    )
    hands = np.concatenate([validated[:, 0:2], validated[:, 2:4]], axis=1)
    print(f"[smoke] hands range = [{hands.min():.3f}, {hands.max():.3f}]")

    policy.reset()
    backend.close()
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
