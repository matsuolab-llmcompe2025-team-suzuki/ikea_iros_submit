"""53D LeRobot GR00T (rotate/insert/flip) → (T,25) smoke (GPU 必要)。

現状 Groot53Backend は desktop 依存 (RAMEN_DESKTOP_REPO)。実行:
    cd /datadrive2/iros_2026_ramen && \
    RAMEN_DESKTOP_REPO=/datadrive2/iros_2026_ramen \
    pixi run -e runtime python /datadrive2/ikea_iros_submit/components/ramen/smoke_groot_53d.py \
        [variant]   # 既定 groot_rotate_leg_200k

mock obs (zeros 画像 + 妥当な body_q) で Groot53Backend を1回 forward し、
boundary (T,25) 契約 (inline、DecoupledSink 相当) を通ることを確認する。
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

from components.ramen.policy import GrootPickTaskspacePolicy, Groot53Backend


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
    variant = sys.argv[1] if len(sys.argv) > 1 else "groot_rotate_leg_200k"
    print(f"[smoke] building Groot53Backend variant={variant} ...")
    t0 = time.monotonic()
    backend = Groot53Backend(variant=variant, task="rotate the table leg to tighten")
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
        "prompt": "rotate the table leg to tighten",
        "t": 0.0,
    }

    t1 = time.monotonic()
    actions = policy.act(obs)["actions"]
    dt = (time.monotonic() - t1) * 1e3
    print(f"[smoke] act() -> actions {actions.shape} {actions.dtype} in {dt:.0f}ms")

    v = _validate_chunk(actions)
    print("[smoke] (T,25) boundary contract PASSED (inline)")
    print(f"[smoke] left_ee_pos[0]  = {v[0, 4:7]}")
    print(f"[smoke] right_ee_pos[0] = {v[0, 11:14]}")
    print(f"[smoke] left_quat|norm| = {float(np.linalg.norm(v[0, 7:11])):.5f}")
    hands = np.concatenate([v[:, 0:2], v[:, 2:4]], axis=1)
    print(f"[smoke] hands range = [{hands.min():.3f}, {hands.max():.3f}]")

    policy.reset()
    backend.close()
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
