"""Pinned organizer adapter with fake robot I/O; no SDK, ROS or sockets.

The scheduled targets pass through the real decoder, clamps and keepalive.
The follower remains a first-order software model, not physical WBC dynamics.
"""

from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REVISION = "497f3ab93e5baa706311daebd31c7a9798258450"


class OfficialFollower:
    def __init__(self, root, gravity):
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision != REVISION:
            raise ValueError("Unexpected organizer revision")
        sys.path.insert(0, str(Path(root) / "reference/wbc_adapter/tests"))
        import test_joint_lane as official

        self.official = official
        self.ctx = official.make_ctx(dex1=official.FakeDex1())
        self.q = np.zeros(14)
        self.gravity = gravity
        self.commanded = self.q.copy()
        self.times = np.array([time.monotonic()])
        self.targets = self.q[None].copy()
        self.hand_target = np.full(2, 4.5)
        self._update_state()

    def _update_state(self):
        self.ctx.backend.q[15:22] = self.q[:7]
        self.ctx.backend.q[29:36] = self.q[7:]

    def _consume_goal(self):
        if self.ctx.backend.goals:
            goal = self.ctx.backend.goals[-1]
            self.ctx.backend.goals.clear()
            upper = np.asarray(goal["target_upper_body_pose"])
            self.targets = np.vstack([self.commanded, np.c_[upper[:, :7], upper[:, 14:21]]])
            self.times = np.r_[time.monotonic(), goal["target_time"]]
        if self.ctx.dex1_pub.sent:
            hand = self.ctx.dex1_pub.sent[-1]
            self.ctx.dex1_pub.sent.clear()
            self.hand_target = (1 - np.array([hand["left"], hand["right"]])) * 2.65

    def packet(self, message):
        self._update_state()
        if message.startswith(b"goto"):
            self.official.wbc_driver._handle_goto(self.ctx, message)
        elif message.startswith(b"joint"):
            self.official.wbc_driver._handle_joint(self.ctx, message)
        else:
            raise ValueError("Unexpected wire topic")
        self._consume_goal()

    def step(self, now, dt):
        if now - self.ctx.last_publish_time >= 0.2:
            self.official.wbc_driver._publish_keepalive(self.ctx)
            self._consume_goal()
        self.commanded = np.array([
            np.interp(now, self.times, self.targets[:, j]) for j in range(14)
        ])
        desired = self.commanded.copy()
        if self.gravity is not None:
            _, offset = self.gravity.apply(self.q)
            desired -= offset
        self.q += np.clip((desired - self.q) * min(1.0, dt / 0.12), -0.8 * dt, 0.8 * dt)
        self._update_state()
        return self.q

    def report(self):
        stats = self.ctx.stats
        return {"organizer_revision": REVISION, "messages": stats.messages,
                "rejected": stats.rejected, "stale": stats.stale,
                "goto_accepted": stats.goto_accepted, "goto_rejected": stats.goto_rejected,
                "summary": stats.report(), "physics_validated": False}
