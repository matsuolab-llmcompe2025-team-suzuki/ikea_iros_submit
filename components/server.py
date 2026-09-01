#!/usr/bin/env python3
"""THE FILE WE REPLACE — our policy, running on the Jetson AGX Thor.

    Thor  192.168.100.1   JetPack 7 · CUDA 13.0 · Python 3.12 · sm_110
    Orin  192.168.100.2   runs components/client.py

This reference emits a hold-still action so the whole pipeline runs before
our model exists. Swap the body of :meth:`act` for real inference and
adjust :attr:`metadata` to match. Nothing else in the repo needs to change.

    # on the Thor
    python components/server.py --lane sonic --port 8765

Lane picks the action space:

    sonic      motion_token (T, 64) + left/right_hand_joints (T, 7)
               |motion_token| must stay <= 1.25
    decoupled  actions (T, 25) task-space; see boundary/actions.py for the
               row layout

THOR SETUP, THE PART THAT BITES: a plain `uv sync` installs the dGPU torch
build (sm_80/90/100/120) and every kernel launch dies with "no kernel image
available" on Thor's sm_110. Use the Thor install path for our framework.
If we are on Isaac-GR00T that means scripts/deployment/thor/install_deps.sh
followed by `source scripts/activate_thor.sh`, and `git lfs install &&
git lfs pull` or the checkpoints stay as pointer files.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.transport import serve_policy  # noqa: E402

LANES = ("sonic", "decoupled")


class Policy:
    """Reference policy: correct shapes, no intelligence.

    Replace the guts. Keep the three members — ``metadata``, ``act`` and
    ``reset`` — because ``client.py`` calls exactly those.
    """

    ACTION_CHUNK = 16     # rows returned per inference
    OBS_CHUNK = 1         # frames of history we want per observation

    def __init__(self, lane: str, delay_ms: float = 0.0):
        if lane not in LANES:
            raise ValueError(f"unknown lane {lane!r}; expected one of {LANES}")
        self.lane = lane
        self._delay_s = delay_ms / 1000.0
        self._steps = 0
        # >>> load our model here <<<
        # self.model = MyVLA.from_pretrained(...).to("cuda").eval()

    @property
    def metadata(self) -> dict:
        """Announced to the client on connect, before any observation.

        The client uses this to size its buffers and to check that the
        server it reached is the one it expected, so keep it honest.
        """
        return {
            "lane": self.lane,
            "action_chunk_size": self.ACTION_CHUNK,
            "obs_chunk_size": self.OBS_CHUNK,
            # Declare only what we actually consume; the client will not
            # spend time encoding images we ignore.
            "camera_keys": ["ego_view"],
            "wants_state": True,
            "wants_prompt": True,
        }

    def act(self, obs: dict) -> dict:
        """One inference step.

        ``obs`` arrives from client.py as:
            images   {key: (480, 640, 3) uint8 RGB}   only our camera_keys
            body_q   (29,) float32   canonical G1 joints, radians
            base_quat(4,)  float32   wxyz
            prompt   str             the task instruction
            t        float           Orin capture timestamp
        """
        self._steps += 1
        T = self.ACTION_CHUNK
        if self._delay_s:
            time.sleep(self._delay_s)   # stand-in for real inference time

        # >>> our inference goes here <<<
        # images = obs["images"]["ego_view"]      # (480, 640, 3) uint8 RGB
        # state  = obs["body_q"]                  # (29,) float32
        # chunk  = self.model.infer(images, state, obs["prompt"])

        if self.lane == "sonic":
            return {
                "motion_token": np.zeros((T, 64), dtype=np.float32),
                "left_hand_joints": np.zeros((T, 7), dtype=np.float32),
                "right_hand_joints": np.zeros((T, 7), dtype=np.float32),
            }

        # Task-space: zeros everywhere except the quaternions, which must be
        # unit length or boundary/actions.py rejects the chunk.
        actions = np.zeros((T, 25), dtype=np.float32)
        actions[:, 7] = 1.0     # left  end-effector quat w
        actions[:, 14] = 1.0    # right end-effector quat w
        return {"actions": actions}

    def reset(self) -> dict:
        """Called once at the start of every attempt. Drop episode state."""
        self._steps = 0
        # >>> clear our action history / KV cache / observation buffer <<<
        return {"ok": True}


def _build_policy(lane: str, delay_ms: float):
    """RAMEN_POLICY env で policy 実装を選ぶ。

    未設定 / "stub"        → 参照 hold-still Policy (conformance の既定、挙動不変)。
    "groot_pick"           → GrootPickTaskspacePolicy + HoldPoseBackend (GPU 不要 PoC)。
    "groot_pick_real"      → 同上 + 実 GR00T pick (38D Isaac native、self-contained worker)。
    "groot_53d_real"       → 同上 + 実 53D LeRobot GR00T。skill は env RAMEN_VARIANT で選ぶ
                             (例 groot_rotate_leg_200k / groot_insert_leg_200k /
                              groot_overlay / groot_flip_table_n17_2_baseline)。desktop 依存。
    """
    choice = os.environ.get("RAMEN_POLICY", "stub").strip().lower()
    if choice in ("", "stub", "reference"):
        return Policy(lane, delay_ms)
    if choice in ("groot_pick", "groot_pick_real", "groot_53d_real"):
        if lane != "decoupled":
            raise SystemExit(f"RAMEN_POLICY={choice} requires --lane decoupled")
        from components.ramen.policy import GrootPickTaskspacePolicy

        backend = None
        if choice == "groot_pick_real":
            from components.ramen.policy import GrootWorkerBackend

            print("[server] loading real GR00T pick worker (may take ~1 min)")
            backend = GrootWorkerBackend()
        elif choice == "groot_53d_real":
            from components.ramen.policy import Groot53Backend

            variant = os.environ.get("RAMEN_VARIANT", "").strip()
            if not variant:
                raise SystemExit(
                    "RAMEN_POLICY=groot_53d_real requires RAMEN_VARIANT "
                    "(e.g. groot_rotate_leg_200k / groot_insert_leg_200k / "
                    "groot_overlay / groot_flip_table_n17_2_baseline)"
                )
            print(f"[server] loading real 53D GR00T ({variant}) ...")
            backend = Groot53Backend(variant=variant)
        print(f"[server] using GrootPickTaskspacePolicy (decoupled, {choice})")
        return GrootPickTaskspacePolicy(lane=lane, backend=backend)
    if choice == "groot_orchestrator":
        if lane != "decoupled":
            raise SystemExit("RAMEN_POLICY=groot_orchestrator requires --lane decoupled")
        from components.ramen.policy import OrchestratorTaskspacePolicy

        print("[server] using OrchestratorTaskspacePolicy (full orchestrator, YOLO+5 skill)")
        return OrchestratorTaskspacePolicy(lane=lane)
    raise SystemExit(
        f"unknown RAMEN_POLICY={choice!r}; expected stub / groot_pick / "
        "groot_pick_real / groot_53d_real / groot_orchestrator"
    )


# serve 前に実 GR00T 推論を1度回す modes (cold CUDA kernel compile / graph capture を
# 逃がす)。stub / GPU 無し HoldPose は対象外。IAC eval 指摘: 初回 .act() が ~10.9s かかり
# 実 run 最初の act_policy stage で stale-chunk window を食う。MEL-CRAFT も同パターン。
_WARMUP_MODES = ("groot_pick_real", "groot_53d_real", "groot_orchestrator")


def _warmup_policy(policy, label: str, iters: int = 2) -> None:
    """Dummy obs で act() を数回叩き、GR00T の初回 compile を serve 前に済ませる。

    best-effort: 失敗しても serve は継続する (warmup が理由でサーバを落とさない)。
    dummy obs は boundary の obs 形式 (body_q (29,) + ego/wrist 画像 (480,640,3) uint8)。
    orchestrator は initial_skill=rotate_table_base で起動するので、zeros 画像でも
    最初の skill worker が dispatch されて温まる (perception 検出は skill 遷移用で、
    現 active skill の GR00T は毎 tick 走る)。
    """
    dummy = {
        "body_q": np.zeros((29,), dtype=np.float32),
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "images": {
            "ego_view": np.zeros((480, 640, 3), dtype=np.uint8),
            "left_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            "right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        },
        "prompt": "warmup",
    }
    t0 = time.time()
    try:
        for i in range(iters):
            policy.act(dummy)
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()   # warmup tick の state を捨てて本番を綺麗に始める
        print(f"[server] warmup done ({label}, {iters} iters, {time.time() - t0:.1f}s)")
    except Exception as exc:   # noqa: BLE001 — warmup 失敗で serve を止めない
        print(f"[server] warmup skipped ({label}): {type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", choices=LANES,
                        default=os.environ.get("PEVAL_LANE", "sonic"),
                        help="Action space. Must match our manifest.")
    parser.add_argument("--delay-ms", type=float, default=0.0,
                        help="Fake inference time. Use it to see how our client "
                             "behaves at realistic latency before our model exists.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"[server] lane={args.lane} delay={args.delay_ms:.0f}ms")
    policy = _build_policy(args.lane, args.delay_ms)
    choice = os.environ.get("RAMEN_POLICY", "stub").strip().lower()
    if choice in _WARMUP_MODES and os.environ.get("RAMEN_WARMUP", "1") != "0":
        print(f"[server] warming up ({choice}) before serving ...")
        _warmup_policy(policy, choice)
    serve_policy(policy, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
