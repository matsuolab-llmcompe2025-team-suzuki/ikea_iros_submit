"""full orchestrator driver smoke (GPU 必要)。build + 数 tick で (T,25) を検証。

実行:
    cd /datadrive2/iros_2026_ramen && \
    RAMEN_WORKER_PYTHON=.../model/subtask_policy_training/.venv/bin/python \
    RAMEN_WORKER_PYTHON_53D=.../inference/desktop/.pixi/envs/default/bin/python \
    pixi run -e runtime python /datadrive2/ikea_iros_submit/components/ramen/smoke_orchestrator.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from components.ramen.orchestrator_driver import OrchestratorDriver


def _validate(arr: np.ndarray) -> None:
    assert arr.ndim == 2 and arr.shape[1] == 25, f"bad shape {arr.shape}"
    assert np.all(np.isfinite(arr))
    hands = np.concatenate([arr[:, 0:2], arr[:, 2:4]], axis=1)
    assert np.max(np.abs(hands)) <= 1.0 + 1e-3
    for sl in (slice(7, 11), slice(14, 18)):
        assert np.all(np.abs(np.linalg.norm(arr[:, sl], axis=1) - 1.0) <= 1e-2)


def main() -> int:
    print("[smoke] building OrchestratorDriver (perception + lazy skills) ...")
    t0 = time.monotonic()
    drv = OrchestratorDriver()
    print(f"[smoke] built in {time.monotonic() - t0:.1f}s")

    obs = {
        "images": {
            "ego_view": np.zeros((480, 640, 3), dtype=np.uint8),
            "left_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            "right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        },
        "body_q": np.zeros(29, dtype=np.float32),
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "prompt": "assemble the table",
        "t": 0.0,
    }
    for i in range(8):
        out = drv.act(obs)
        actions = out["actions"]
        _validate(actions)
        print(f"[smoke] tick {i}: actions {actions.shape} skill={out['current_skill']} "
              f"left_ee={np.round(actions[0,4:7],3)}")
    drv.close()
    print("[smoke] OK — orchestrator drove ticks and produced valid (T,25)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
