"""Dex1 を目標の開度へ動かす skill (Issue #141 束 1-8 / D2)。

# なぜ要るか

評価経路は skill を始める前に「手を全開にする → 腕を開始姿勢へ → 手を教師の frame 0 の
開度へ」を必ず通っていたのに、本番経路は手を前の状態のまま policy を始めていた
(INF-14)。学習データの frame 0 は必ず決まった開度から始まるので、開始の手の開度が
違うと最初の chunk から入力が学習と違う。

その手順を「腕を保持したまま手だけ動かす」skill にして、頭の手順
(手を開く → 腕の pre-motion → 手を開始の開度へ → N 秒保持) の 1 つとして
評価でも本番でも同じものを使う。

# 3 つの目標

| target | 目標の開度 | 接触で止まってよいか |
|---|---|---|
| `open` | 全開 (`HAND_GRIP_MAX` = 5.4 rad) | いいえ |
| `pose` | その skill の教師 frame 0 の開度 (`initial_pose.dex1_opening_fraction` × 4.5) | いいえ |
| `grasp` | 同上 | **はい** (脚を掴んで止まる。力センサが無いので「動かなくなった位置」で判定) |

数値は `skill_config.yaml` の `hand_pre_motion` section (評価経路の実績値)。
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Optional

import numpy as np

from inference.desktop.lower_policy.actuators.hand import HAND_GRIP_MAX
from inference.desktop.lower_policy.initial_pose import initial_pose_from_config
from inference.desktop.lower_policy.pose_utils import arm_positions_from_joint_state
from inference.desktop.lower_policy.skills.base import Skill
from inference.desktop.lower_policy.skills.hand_ramp import (
    STABLE_HISTORY_LEN,
    build_hand_target_ramp,
    hand_target_completion_mode,
)

TARGET_MODES = ("open", "pose", "grasp")


def _measured_arm_14(obs: dict) -> np.ndarray:
    """obs の joint_state から腕 14 関節を取る (名前で引く)。"""
    state = obs.get("joint_state")
    if state is None:
        raise RuntimeError("hand pre-motion requires a live joint state to hold the arms")
    return arm_positions_from_joint_state(
        tuple(state.name), np.asarray(state.position, dtype=np.float64), "hand pre-motion"
    )


class HandPreMotionSkill(Skill):
    """腕を保持したまま Dex1 を目標の開度へ動かす。

    Args:
        name: skill 名 (dispatcher の key)。
        hand_actuator: `send_action([left, right])` を持つもの。
        target_rad: 目標の開度 (left, right) [rad]。
        allow_closing_contact: 閉じる途中で接触して安定したら完了にしてよいか。
        velocity_limit_rad_s / tolerance_rad / required_stable_samples: ramp と到達の判定。
        minimum_timeout_sec / timeout_margin_sec: 締め切り = max(minimum, 所要時間 + margin)。
        state_max_age_s: これより古い Dex1 の state は使わない。
        control_hz: ramp を刻む周期 (tick の周期)。
    """

    def __init__(
        self,
        name: str,
        *,
        hand_actuator: Any,
        target_rad: tuple[float, float],
        allow_closing_contact: bool = False,
        velocity_limit_rad_s: float = 1.5,
        tolerance_rad: float = 0.15,
        required_stable_samples: int = 5,
        minimum_timeout_sec: float = 5.0,
        timeout_margin_sec: float = 3.0,
        state_max_age_s: float = 0.5,
        control_hz: float = 30.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        target = np.asarray(target_rad, dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError(f"target_rad must be finite 2-D, got {target_rad!r}")
        for label, value in (
            ("velocity_limit_rad_s", velocity_limit_rad_s),
            ("tolerance_rad", tolerance_rad),
            ("minimum_timeout_sec", minimum_timeout_sec),
            ("state_max_age_s", state_max_age_s),
            ("control_hz", control_hz),
        ):
            if value <= 0.0:
                raise ValueError(f"{label} must be > 0, got {value}")
        if required_stable_samples <= 0:
            raise ValueError(
                f"required_stable_samples must be > 0, got {required_stable_samples}"
            )
        self.name = name
        self._hand_actuator = hand_actuator
        self._target = target
        self._allow_closing_contact = bool(allow_closing_contact)
        self._velocity_limit_rad_s = float(velocity_limit_rad_s)
        self._tolerance_rad = float(tolerance_rad)
        self._required_stable_samples = int(required_stable_samples)
        self._minimum_timeout_sec = float(minimum_timeout_sec)
        self._timeout_margin_sec = float(timeout_margin_sec)
        self._state_max_age_s = float(state_max_age_s)
        self._control_hz = float(control_hz)
        self._time_fn = time_fn
        self._reset()

    @classmethod
    def from_config(
        cls,
        skill_config: dict,
        skill_name: str,
        *,
        target: str,
        hand_actuator: Any,
        name: Optional[str] = None,
        **overrides: Any,
    ) -> "HandPreMotionSkill":
        """`skill_config.yaml` から作る。

        Args:
            skill_config: `yaml.safe_load(skill_config.yaml)` の結果。
            skill_name: 開度の出どころになる学習 skill (`skills.<name>.initial_pose`)。
            target: `open` / `pose` / `grasp`。
            name: skill 名 (既定は `hand_<target>_<skill_name>`)。
        """
        if target not in TARGET_MODES:
            raise ValueError(f"target must be one of {TARGET_MODES}, got {target!r}")
        settings = dict(skill_config.get("hand_pre_motion") or {})
        unknown = sorted(
            set(settings)
            - {
                "velocity_limit_rad_s", "tolerance_rad", "required_stable_samples",
                "minimum_timeout_sec", "timeout_margin_sec", "state_max_age_s",
            }
        )
        if unknown:
            raise ValueError(f"hand_pre_motion has unknown keys: {unknown}")
        if target == "open":
            target_rad = (HAND_GRIP_MAX, HAND_GRIP_MAX)
        else:
            target_rad = initial_pose_from_config(skill_config, skill_name).dex1_target_rad
        settings.update(overrides)
        return cls(
            name or f"hand_{target}_{skill_name}",
            hand_actuator=hand_actuator,
            target_rad=target_rad,
            allow_closing_contact=(target == "grasp"),
            **settings,
        )

    # ---- lifecycle ----

    def _reset(self) -> None:
        self._hold_arm: Optional[np.ndarray] = None
        self._ramp: tuple[np.ndarray, ...] = ()
        self._ramp_index = 0
        self._start_measured: Optional[np.ndarray] = None
        self._history: list[np.ndarray] = []
        self._stable_samples = 0
        self._deadline: Optional[float] = None
        self._complete = False
        self._completion_mode: Optional[str] = None
        self._failure_reason: Optional[str] = None

    def _on_start(self, params: dict) -> None:
        self._reset()
        # 測定値が来るまで ramp を始められないので、まず「待つ期限」を張る。
        # 測定が来たら _begin_ramp が移動時間に応じた期限に張り替える。これが無いと
        # Dex1 の state が途切れたときに完了も失敗もせず、腕を持ったまま止まらない。
        self._deadline = self._time_fn() + self._minimum_timeout_sec

    def _on_stop(self) -> None:
        pass

    # ---- per-tick ----

    def step(self, obs: dict) -> np.ndarray:
        if self._hold_arm is None:
            self._hold_arm = _measured_arm_14(obs)
        measured = self._measured_hand(obs)
        if measured is not None and self._start_measured is None:
            self._begin_ramp(measured)
        if self._ramp:
            command = self._ramp[min(self._ramp_index, len(self._ramp) - 1)]
            self._ramp_index += 1
            self._hand_actuator.send_action(command)
        if measured is not None and not self._complete:
            self._update_completion(measured)
        if not self._complete:
            self._check_deadline()
        return self._hold_arm.copy()

    def _measured_hand(self, obs: dict) -> Optional[np.ndarray]:
        """新しくて有限な Dex1 の開度。古い / 未受信なら None。"""
        state = obs.get("hand_state")
        if state is None:
            return None
        now_ns = time.monotonic_ns()
        age_s = max(
            now_ns - int(state.left_received_monotonic_ns),
            now_ns - int(state.right_received_monotonic_ns),
        ) / 1e9
        measured = np.asarray(state.position_rad, dtype=np.float64)
        if age_s > self._state_max_age_s or measured.shape != (2,) or not np.isfinite(measured).all():
            self._history.clear()
            self._stable_samples = 0
            return None
        return measured

    def _begin_ramp(self, measured: np.ndarray) -> None:
        self._start_measured = measured.copy()
        self._ramp = build_hand_target_ramp(
            measured,
            self._target,
            command_hz=self._control_hz,
            velocity_limit_rad_s=self._velocity_limit_rad_s,
        )
        self._ramp_index = 0
        travel_s = float(np.max(np.abs(self._target - measured))) / self._velocity_limit_rad_s
        self._deadline = self._time_fn() + max(
            self._minimum_timeout_sec, travel_s + self._timeout_margin_sec
        )
        print(
            f"[hand] {self.name}: {measured.tolist()} -> {self._target.tolist()} "
            f"({len(self._ramp)} steps)",
            file=sys.stderr,
        )

    def _update_completion(self, measured: np.ndarray) -> None:
        self._history.append(measured.copy())
        self._history = self._history[-STABLE_HISTORY_LEN:]
        mode = hand_target_completion_mode(
            start=self._start_measured,
            target=self._target,
            measured_history=self._history,
            tolerance_rad=self._tolerance_rad,
            allow_closing_contact=self._allow_closing_contact,
        )
        self._stable_samples = self._stable_samples + 1 if mode == "target" else 0
        if self._stable_samples >= self._required_stable_samples or mode == "closing_contact":
            self._complete = True
            self._completion_mode = mode
            print(
                f"[hand] {self.name}: {mode} (measured={measured.tolist()}, "
                f"residual={np.abs(measured - self._target).tolist()})",
                file=sys.stderr,
            )

    def _check_deadline(self) -> None:
        if self._deadline is None or self._time_fn() < self._deadline:
            return
        measured = "unknown" if not self._history else self._history[-1].tolist()
        self._failure_reason = (
            f"{self.name} did not reach {self._target.tolist()} before the deadline "
            f"(measured={measured})"
        )

    # ---- orchestrator が読む状態 ----

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def failure_reason(self) -> Optional[str]:
        """締め切りを過ぎたら理由が入る (orchestrator が安全停止に回す)。"""
        return self._failure_reason

    @property
    def completion_mode(self) -> Optional[str]:
        """完了の仕方 (`target` / `closing_contact`)。記録用。"""
        return self._completion_mode
