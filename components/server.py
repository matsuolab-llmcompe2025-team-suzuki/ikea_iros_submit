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
    serve_policy(Policy(args.lane, args.delay_ms), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
