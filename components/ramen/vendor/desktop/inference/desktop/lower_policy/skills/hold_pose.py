"""開始姿勢のまま N 秒待つ skill (Issue #141 束 1-8 / D2)。

頭の手順 (手を開く → 腕の pre-motion → 手を開始の開度へ → **N 秒保持**) の最後。
本番は stage の境界の Enter を無くして自然に進むので、その代わりに「開始姿勢に着いて
から少し待つ」時間をここで作る (ユーザー決定 2026-09-14)。N は起動引数で指定する
(評価は Enter があるので既定 0 秒、本番は 3 秒)。

腕は今の測定値を保持するだけで、手にも腰にも指令を出さない。
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Optional

import numpy as np

from inference.desktop.lower_policy.pose_utils import arm_positions_from_joint_state
from inference.desktop.lower_policy.skills.base import Skill


class HoldPoseSkill(Skill):
    """最初の tick の腕の姿勢を `hold_sec` 秒のあいだ出し続ける。

    Args:
        hold_sec: 保持する秒数。0 なら最初の tick で完了する。
        name: skill 名 (dispatcher の key)。
    """

    def __init__(
        self,
        hold_sec: float,
        *,
        name: str = "hold_pose",
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        if hold_sec < 0.0:
            raise ValueError(f"hold_sec must be >= 0, got {hold_sec}")
        self.name = name
        self._hold_sec = float(hold_sec)
        self._time_fn = time_fn
        self._hold_arm: Optional[np.ndarray] = None
        self._started_at: Optional[float] = None
        self._complete = False

    def _on_start(self, params: dict) -> None:
        self._hold_arm = None
        self._started_at = None
        self._complete = False

    def _on_stop(self) -> None:
        pass

    def step(self, obs: dict) -> np.ndarray:
        if self._hold_arm is None:
            state = obs.get("joint_state")
            if state is None:
                raise RuntimeError(f"{self.name} requires a live joint state to hold")
            self._hold_arm = arm_positions_from_joint_state(
                tuple(state.name),
                np.asarray(state.position, dtype=np.float64),
                self.name,
            )
            self._started_at = self._time_fn()
            print(f"[hold] {self.name}: holding for {self._hold_sec:g}s", file=sys.stderr)
        if not self._complete and self._time_fn() - self._started_at >= self._hold_sec:
            self._complete = True
        return self._hold_arm.copy()

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def hold_sec(self) -> float:
        return self._hold_sec
