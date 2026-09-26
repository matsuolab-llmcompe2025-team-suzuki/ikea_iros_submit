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
        completion_ready_fn: Optional[Callable[[], bool]] = None,
        completion_error_fn: Optional[Callable[[], Optional[str]]] = None,
        completion_description: str = "external readiness gate",
        completion_timeout_s: Optional[float] = None,
    ) -> None:
        super().__init__()
        if hold_sec < 0.0:
            raise ValueError(f"hold_sec must be >= 0, got {hold_sec}")
        self.name = name
        self._hold_sec = float(hold_sec)
        self._time_fn = time_fn
        self._completion_ready_fn = completion_ready_fn
        self._completion_error_fn = completion_error_fn
        self._completion_description = str(completion_description)
        # None = 時間制限なし (読み込み中の model は待つしかない。止めるのは人)。
        self._completion_timeout_s = (
            None if completion_timeout_s is None else float(completion_timeout_s)
        )
        if self._completion_timeout_s is not None and self._completion_timeout_s <= 0.0:
            raise ValueError("completion_timeout_s must be > 0")
        self._hold_arm: Optional[np.ndarray] = None
        self._started_at: Optional[float] = None
        self._complete = False
        self._waiting_logged = False
        self._failure_reason: Optional[str] = None

    def set_completion_gate(
        self,
        ready_fn: Callable[[], bool],
        *,
        error_fn: Optional[Callable[[], Optional[str]]] = None,
        description: str,
        timeout_s: Optional[float] = None,
    ) -> None:
        """model の載せ替えが終わるまで frame-zero を保持させる。

        保持が最短秒数で終わってしまうと、まだ読み込み中の policy に切り替わる。
        `ModelResidency` の読み込み完了をここで待たせることで、腕・手の target を
        生かしたまま載せ替えを終わらせる。読み込みの失敗 (`error_fn`) は故障として
        止めるが、読み込み中は時間制限なしで待つ (`timeout_s=None`、既定)。
        """
        if timeout_s is not None and timeout_s <= 0.0:
            raise ValueError("completion gate timeout_s must be > 0")
        self._completion_ready_fn = ready_fn
        self._completion_error_fn = error_fn
        self._completion_description = str(description)
        self._completion_timeout_s = None if timeout_s is None else float(timeout_s)

    def _on_start(self, params: dict) -> None:
        self._hold_arm = None
        self._started_at = None
        self._complete = False
        self._waiting_logged = False
        self._failure_reason = None

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
        elapsed = self._time_fn() - self._started_at
        if not self._complete and elapsed >= self._hold_sec:
            error = (
                self._completion_error_fn()
                if self._completion_error_fn is not None
                else None
            )
            if error is not None:
                self._failure_reason = (
                    f"{self.name} cannot continue because "
                    f"{self._completion_description} failed: {error}"
                )
            elif self._completion_ready_fn is None or self._completion_ready_fn():
                self._complete = True
                if self._completion_ready_fn is not None:
                    print(
                        f"[hold] {self.name}: "
                        f"{self._completion_description} ready; continuing",
                        file=sys.stderr,
                    )
            elif (
                self._completion_timeout_s is not None
                and elapsed >= self._completion_timeout_s
            ):
                self._failure_reason = (
                    f"{self.name} timed out after "
                    f"{self._completion_timeout_s:g}s waiting for "
                    f"{self._completion_description}; held pose remains active"
                )
            elif not self._waiting_logged:
                print(
                    f"[hold] {self.name}: frame-zero pose reached; holding while "
                    f"{self._completion_description}",
                    file=sys.stderr,
                )
                self._waiting_logged = True
        return self._hold_arm.copy()

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def failure_reason(self) -> Optional[str]:
        return self._failure_reason

    @property
    def hold_sec(self) -> float:
        return self._hold_sec
