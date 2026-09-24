"""Post-walk collision-aware arm staging for the real Phase 1 profiles.

The robot walks with its arms lowered: stage 0 first lowers them to
``skills.walk_lowered_pose`` and then holds the measured pose.
Only after the base has stopped do the arms follow the proven evaluation path:
shoulders backward, laterally outside the table, forward while still outside,
then the learned-policy start pose.  No waist, hand, or walking command is
issued by these skills.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from inference.desktop.lower_policy.actuators.base import WalkActuator
from inference.desktop.lower_policy.actuators.g1_arm_sdk import (
    G1_ARM_POSITION_LOWER_RAD,
    G1_ARM_POSITION_UPPER_RAD,
)
from inference.desktop.lower_policy.skills.base import Skill


ARM_INDICES = np.arange(15, 29, dtype=np.int64)
LEG_INDICES = np.arange(0, 12, dtype=np.int64)
SHOULDER_PITCH_INDICES = (0, 7)
SHOULDER_ROLL_INDICES = (1, 8)
ELBOW_INDICES = (3, 10)
LOWERED_WALK_TRACKING_MARGIN_RAD = 0.05
# 歩行してよい腕の範囲 (関節群ごとの絶対値の上限 [rad])。
_LOWERED_WALK_LIMITS = {
    "shoulder_pitch": (SHOULDER_PITCH_INDICES, 0.75),
    "shoulder_roll": (SHOULDER_ROLL_INDICES, 0.65),
    "elbow": (ELBOW_INDICES, 1.00),
}
ARM_JOINT_NAMES = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)


def _measured_arm(obs: dict) -> np.ndarray:
    state = obs.get("joint_state")
    positions = None if state is None else getattr(state, "position", None)
    array = np.asarray(positions, dtype=np.float64) if positions is not None else None
    if array is None or array.shape != (29,) or not np.isfinite(array).all():
        raise RuntimeError("live finite 29-D joint state is required for arm staging")
    return array[ARM_INDICES].copy()


def _measured_arm_velocity(obs: dict) -> np.ndarray:
    state = obs.get("joint_state")
    velocities = None if state is None else getattr(state, "velocity", None)
    array = np.asarray(velocities, dtype=np.float64) if velocities is not None else None
    if array is None or array.shape != (29,) or not np.isfinite(array).all():
        raise RuntimeError(
            "live finite 29-D joint velocity is required for arm staging"
        )
    return array[ARM_INDICES].copy()


def _measured_max_leg_speed(obs: dict) -> float:
    state = obs.get("joint_state")
    velocities = None if state is None else getattr(state, "velocity", None)
    array = np.asarray(velocities, dtype=np.float64) if velocities is not None else None
    if array is None or array.shape != (29,) or not np.isfinite(array).all():
        raise RuntimeError(
            "live finite 29-D joint velocity is required before arm pre-motion"
        )
    return float(np.max(np.abs(array[LEG_INDICES])))


def lowered_walk_pose_violation(arm: Sequence[float]) -> str | None:
    """下がった範囲から外れている関節を返す (外れていなければ None)。

    `validate_lowered_walk_pose` と同じ判定を、例外にせず文字列で返す。
    stage 0 は「外れていたら下ろしてから歩く」ので、判定と中断を分ける。
    """
    values = np.asarray(arm, dtype=np.float64)
    if values.shape != (14,) or not np.isfinite(values).all():
        return "walk arm pose must be finite 14-D"
    for label, (indices, maximum) in _LOWERED_WALK_LIMITS.items():
        actual = float(np.max(np.abs(values[list(indices)])))
        allowed = maximum + LOWERED_WALK_TRACKING_MARGIN_RAD
        if actual > allowed:
            return (
                f"{label}={actual:.6f}rad > {allowed:.3f}rad "
                f"(nominal={maximum:.3f}rad, tracking_margin="
                f"{LOWERED_WALK_TRACKING_MARGIN_RAD:.3f}rad)"
            )
    return None


def validate_lowered_walk_pose(arm: Sequence[float]) -> np.ndarray:
    """Reject an arm pose outside the lowered walking corridor.

    The nominal bounds describe the commanded corridor, while the explicit
    0.05 rad margin covers measured tracking error and encoder quantisation.
    Comparing directly against the nominal bound caused a safe elbow reading
    displayed as 1.000 rad to fail because of a sub-display-precision excess.
    """

    values = np.asarray(arm, dtype=np.float64)
    if values.shape != (14,) or not np.isfinite(values).all():
        raise ValueError("walk arm pose must be finite 14-D")
    for label, (indices, maximum) in _LOWERED_WALK_LIMITS.items():
        actual = float(np.max(np.abs(values[list(indices)])))
        allowed = maximum + LOWERED_WALK_TRACKING_MARGIN_RAD
        if actual > allowed:
            raise RuntimeError(
                f"arms are not in the lowered walk envelope: {label}="
                f"{actual:.6f}rad > {allowed:.3f}rad "
                f"(nominal={maximum:.3f}rad, tracking_margin="
                f"{LOWERED_WALK_TRACKING_MARGIN_RAD:.3f}rad); walking/arm "
                "pre-motion is blocked"
            )
    return values.copy()


class MeasuredArmWalkHoldSkill(Skill):
    """歩く前に腕を下ろした姿勢へ動かし、その実測を歩行中ずっと保持する。

    下ろす先 (``lowered_pose_rad``) があるときは、実測がそこから離れていれば
    (どれかの関節が到達判定の許容を超えていれば) 下ろし終わってから保持する。
    歩行の範囲の中でも下ろす: 会場の開始姿勢 (運営 WBC の既定) は範囲の中だが、
    前腕を前に出した姿勢のまま (2026-09-24)。下ろす先が無いときは従来どおり、
    範囲の中ならその姿勢を保持し、外なら止める。
    """

    name = "setup"

    def __init__(
        self,
        *,
        dwell_sec: float = 0.5,
        lowered_pose_rad: Sequence[float] | None = None,
        lowering_settings: dict | None = None,
        measured_convergence_checker: Callable[
            [np.ndarray, np.ndarray, np.ndarray], tuple[bool, str]
        ]
        | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        if not math.isfinite(dwell_sec) or dwell_sec <= 0.0:
            raise ValueError("dwell_sec must be positive and finite")
        self._dwell_sec = float(dwell_sec)
        # 歩く前の腕が範囲の外にあったときに向かう姿勢 (Issue #152)。
        # None なら従来どおり「範囲の外なら止める」。
        self._lowered_pose = (
            None
            if lowered_pose_rad is None
            else np.asarray(lowered_pose_rad, dtype=np.float64)
        )
        if self._lowered_pose is not None:
            if self._lowered_pose.shape != (14,) or not np.isfinite(
                self._lowered_pose
            ).all():
                raise ValueError("lowered_pose_rad must be finite 14-D")
            violation = lowered_walk_pose_violation(self._lowered_pose)
            if violation is not None:
                raise ValueError(
                    f"lowered_pose_rad is itself outside the walk envelope: {violation}"
                )
        self._lowering_settings = dict(lowering_settings or {})
        # 下ろす必要があるかの判定は、下ろし終わりの判定 (pre-motion) と同じ許容で行う。
        self._lowered_tolerance = float(
            self._lowering_settings.get("measured_tolerance_rad", 0.10)
        )
        # 会場は運営 IK が別の関節角で同じ手首の姿勢に届くので、task-space で確かめる。
        self._measured_convergence_checker = measured_convergence_checker
        self._time_fn = time_fn
        self._hold: np.ndarray | None = None
        self._lowering: "CollisionAwareArmPreMotionSkill | None" = None
        self._started_at: float | None = None
        self._latched_at: float | None = None

    def _on_start(self, params: dict) -> None:
        self._hold = None
        self._lowering = None
        self._started_at = self._time_fn()
        self._latched_at = None

    def _on_stop(self) -> None:
        pass

    def _latch(self, measured: np.ndarray) -> np.ndarray:
        self._hold = validate_lowered_walk_pose(measured)
        self._latched_at = self._time_fn()
        print(
            "[setup] measured lowered arm pose latched; walking will keep this pose",
            file=sys.stderr,
        )
        return self._hold.copy()

    def step(self, obs: dict) -> np.ndarray:
        if self._hold is not None:
            return self._hold.copy()
        measured = _measured_arm(obs)
        if self._lowered_pose is None:
            violation = lowered_walk_pose_violation(measured)
            if violation is not None:
                raise RuntimeError(
                    f"arms are not in the lowered walk envelope: {violation}; "
                    "walking/arm pre-motion is blocked"
                )
            return self._latch(measured)
        if self._lowering is None:
            distance = float(np.max(np.abs(measured - self._lowered_pose)))
            if distance <= self._lowered_tolerance:
                return self._latch(measured)
            print(
                f"[setup] arms are {distance:.3f} rad away from walk_lowered_pose; "
                "lowering before the walk",
                file=sys.stderr,
            )
            self._lowering = CollisionAwareArmPreMotionSkill(
                tuple(self._lowered_pose.tolist()),
                skill_name="lower_arms_for_walk",
                waypoint_profile="direct",
                measured_convergence_checker=self._measured_convergence_checker,
                time_fn=self._time_fn,
                **self._lowering_settings,
            )
            self._lowering.start({})
        command = self._lowering.step(obs)
        # 歩く前の腕の退避は安全のための手順なので、時間切れでも歩かずに止める。
        stopped = self._lowering.failure_reason or self._lowering.timeout_reason
        if stopped is not None:
            raise RuntimeError(
                f"lowering the arms before the walk failed: {stopped}"
            )
        if self._lowering.is_complete:
            return self._latch(measured)
        return command

    @property
    def max_dwell_sec(self) -> float | None:
        """下ろし終わるまでは dwell で次へ行かせない (None = dwell 判定を切る)。"""
        if self._latched_at is None:
            return None
        if self._started_at is None:
            return self._dwell_sec
        return (self._latched_at - self._started_at) + self._dwell_sec


class PostWalkArmSettleSkill(Skill):
    """Keep the arms lowered while the commanded walk comes fully to rest.

    ``MoveToTable.stop()`` issues the first zero-velocity command.  This
    separate state reasserts zero velocity on entry, holds a measured lowered
    arm pose for a full settling interval, and reasserts zero once more before
    the dispatcher may enter arm pre-motion.  Keeping this as a distinct skill
    prevents the arm trajectory from starting in the same state transition as
    the walk stop command.
    """

    name = "post_walk_settle"

    def __init__(
        self,
        actuator: WalkActuator,
        *,
        minimum_settle_sec: float = 1.0,
        max_leg_speed_rad_s: float = 0.25,
        required_stable_samples: int = 5,
        timeout_sec: float = 5.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        for label, value in (
            ("minimum_settle_sec", minimum_settle_sec),
            ("max_leg_speed_rad_s", max_leg_speed_rad_s),
            ("timeout_sec", timeout_sec),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive and finite")
        if timeout_sec <= minimum_settle_sec:
            raise ValueError("timeout_sec must exceed minimum_settle_sec")
        if (
            not isinstance(required_stable_samples, int)
            or isinstance(required_stable_samples, bool)
            or required_stable_samples <= 0
        ):
            raise ValueError("required_stable_samples must be a positive integer")
        self._actuator = actuator
        self._minimum_settle_sec = float(minimum_settle_sec)
        self._max_leg_speed = float(max_leg_speed_rad_s)
        self._required_stable_samples = required_stable_samples
        self._timeout_sec = float(timeout_sec)
        self._time_fn = time_fn
        self._hold: np.ndarray | None = None
        self._started_at: float | None = None
        self._stable_samples = 0
        self._last_leg_speed = math.inf
        self._complete = False

    def _send_zero_velocity(self) -> None:
        self._actuator.set_velocity(
            0.0, 0.0, 0.0, duration=max(1.0, self._minimum_settle_sec)
        )

    def _on_start(self, params: dict) -> None:
        self._hold = None
        self._started_at = self._time_fn()
        self._stable_samples = 0
        self._last_leg_speed = math.inf
        self._complete = False
        self._send_zero_velocity()
        print(
            f"[post-walk] zero velocity accepted; holding lowered arms for at "
            f"least {self._minimum_settle_sec:g}s and waiting for measured leg "
            f"speed <= {self._max_leg_speed:g}rad/s",
            file=sys.stderr,
        )

    def _on_stop(self) -> None:
        # The pre-motion transition is allowed only after this second explicit
        # zero command succeeds.  A rejected RPC aborts the transition.
        self._send_zero_velocity()
        print(
            "[post-walk] measured stop confirmed; base zero re-confirmed "
            f"(max_leg_speed={self._last_leg_speed:.4f}rad/s)",
            file=sys.stderr,
        )

    def step(self, obs: dict) -> np.ndarray:
        if self._hold is None:
            self._hold = validate_lowered_walk_pose(_measured_arm(obs))
        assert self._started_at is not None
        now = self._time_fn()
        self._last_leg_speed = _measured_max_leg_speed(obs)
        if self._last_leg_speed <= self._max_leg_speed:
            self._stable_samples += 1
        else:
            self._stable_samples = 0
        elapsed = now - self._started_at
        if (
            elapsed >= self._minimum_settle_sec
            and self._stable_samples >= self._required_stable_samples
        ):
            self._complete = True
        elif elapsed > self._timeout_sec:
            raise TimeoutError(
                "walking did not measurably settle before arm pre-motion: "
                f"elapsed={elapsed:.2f}s, max_leg_speed="
                f"{self._last_leg_speed:.4f}rad/s, threshold="
                f"{self._max_leg_speed:.4f}rad/s"
            )
        return self._hold.copy()

    @property
    def is_complete(self) -> bool:
        return self._complete


@dataclass(frozen=True)
class ArmWaypoint:
    name: str
    target: tuple[float, ...]
    preserve_initial: tuple[int, ...] = ()
    do_not_decrease_from_initial: tuple[int, ...] = ()
    # (関節 index, 加算値)。実測の姿勢からの相対で経由点を作る (Issue #152)。
    # 「今の姿勢のまま少しだけ持ち上げる」を、絶対値を知らずに書くために使う。
    offset_from_initial: tuple[tuple[int, float], ...] = ()

    def resolve(self, initial: np.ndarray) -> np.ndarray:
        target = np.asarray(self.target, dtype=np.float64).copy()
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError(f"waypoint {self.name!r} must be finite 14-D")
        # 実測から値が来た関節 (Issue #159 T4)。腕が既にいる位置なので 1.5 rad の
        # 検査は掛けない。掛けると、rotate_leg の締め途中 (手首 yaw 1.5〜1.61) で
        # 切られた境界が 1 tick 目に ValueError で落ちる。
        from_measured = np.zeros(target.shape, dtype=bool)
        for index in self.preserve_initial:
            target[index] = initial[index]
            from_measured[index] = True
        for index in self.do_not_decrease_from_initial:
            if initial[index] > target[index]:
                target[index] = initial[index]
                from_measured[index] = True
        for index, delta in self.offset_from_initial:
            target[index] = initial[index] + delta
            from_measured[index] = True
        if np.any(np.abs(target[~from_measured]) > 1.5):
            raise ValueError(f"waypoint {self.name!r} exceeds the 1.5rad smoke limit")
        # 自前経路の arm actuator (G1ArmActuator.send_action) と同じ URDF の限界に
        # 収める。大会経路には運営 IK の手前にこの clip が無いので、ここで揃える。
        return np.clip(target, G1_ARM_POSITION_LOWER_RAD, G1_ARM_POSITION_UPPER_RAD)


_PRESERVE_EXCEPT_PITCH = tuple(i for i in range(14) if i not in (0, 7))
_PRESERVE_EXCEPT_PITCH_ROLL_ELBOW = tuple(
    i for i in range(14) if i not in (0, 1, 3, 7, 8, 10)
)


def build_collision_aware_waypoints(final_pose: Sequence[float]) -> tuple[ArmWaypoint, ...]:
    """Build the same clearance order used by the prior real evaluator."""

    final = np.asarray(final_pose, dtype=np.float64)
    if final.shape != (14,) or not np.isfinite(final).all():
        raise ValueError("final policy start pose must be finite 14-D")
    return (
        ArmWaypoint(
            "shoulder_pitch_backward_clearance",
            (0.85, 0, 0, 0, 0, 0, 0, 0.85, 0, 0, 0, 0, 0, 0),
            preserve_initial=_PRESERVE_EXCEPT_PITCH,
            do_not_decrease_from_initial=(0, 7),
        ),
        ArmWaypoint(
            "lateral_high_clearance",
            (0.85, 1.5, 0, 0.4, 0, 0, 0, 0.85, -1.5, 0, 0.4, 0, 0, 0),
            preserve_initial=_PRESERVE_EXCEPT_PITCH_ROLL_ELBOW,
        ),
        ArmWaypoint(
            "forward_outward_clearance",
            (-0.55, 1.5, 0, 0.4, 0, 0, 0, -0.55, -1.5, 0, 0.4, 0, 0, 0),
        ),
        ArmWaypoint("policy_initial_pose", tuple(final.tolist())),
    )


def build_lowering_waypoints(lowered_pose: Sequence[float]) -> tuple[ArmWaypoint, ...]:
    """run の終わりに、起動時の退避を逆にたどって腕を下ろす (Issue #152)。

    腕を上げた状態のまま arm_sdk を手放すと、その姿勢から本体側の制御に戻ることに
    なる。下ろしてから手放せば、次の run は毎回同じ状態から始められる (stage 0 は
    腕が下がった範囲にあることを求める)。経路は `build_collision_aware_waypoints`
    の 3 つの退避姿勢を逆順にたどり、最後に下がった姿勢へ入る。
    """
    lowered = np.asarray(lowered_pose, dtype=np.float64)
    if lowered.shape != (14,) or not np.isfinite(lowered).all():
        raise ValueError("lowered pose must be finite 14-D")
    return (
        ArmWaypoint(
            "return_forward_outward_clearance",
            (-0.55, 1.5, 0, 0.4, 0, 0, 0, -0.55, -1.5, 0, 0.4, 0, 0, 0),
        ),
        ArmWaypoint(
            "return_lateral_high_clearance",
            (0.85, 1.5, 0, 0.4, 0, 0, 0, 0.85, -1.5, 0, 0.4, 0, 0, 0),
        ),
        ArmWaypoint(
            "return_shoulder_pitch_backward_clearance",
            (0.85, 0, 0, 0, 0, 0, 0, 0.85, 0, 0, 0, 0, 0, 0),
        ),
        ArmWaypoint("return_lowered_pose", tuple(lowered.tolist())),
    )


# 肩 pitch の index (左 / 右)。マイナス方向が手先を上げる向き
# (pick の開始姿勢で -0.15 rad → 手先 z 左 +4.0cm / 右 +6.3cm、G1 URDF FK で実測)。
_SHOULDER_PITCH_INDICES = (0, 7)


def build_boundary_lift_waypoints(
    final_pose: Sequence[float], lift_rad: float
) -> tuple[ArmWaypoint, ...]:
    """model の境界で、次の skill の開始姿勢へ「少し上げてから」直接向かう。

    歩行の後に腕を上げる `build_collision_aware_waypoints` (頭上・側方の退避) は、
    **skill 間の移動には広すぎる**。脚を保持した腕を roll ±1.5 まで開くと、持った脚が
    テーブルや反対の腕を薙ぐ。一方で終端姿勢から次の開始姿勢へ直線で向かうと天板を
    擦る可能性がある。そこで肩 pitch だけを `lift_rad` 上げて移り、最後に下ろす。

    4 skill の開始姿勢の手先 z は +0.07〜+0.23 m に収まっており (G1 URDF FK)、
    上下の振れ幅は数 cm で足りる。値は `skill_config.yaml` の
    `arm_pre_motion.boundary_lift_rad` に置く。
    """
    final = np.asarray(final_pose, dtype=np.float64)
    if final.shape != (14,) or not np.isfinite(final).all():
        raise ValueError("final policy start pose must be finite 14-D")
    lift = float(lift_rad)
    if not np.isfinite(lift) or lift < 0.0:
        raise ValueError(f"boundary_lift_rad must be finite and >= 0, got {lift_rad}")
    lifted_final = final.copy()
    for index in _SHOULDER_PITCH_INDICES:
        lifted_final[index] -= lift
    preserve_except_pitch = tuple(
        i for i in range(14) if i not in _SHOULDER_PITCH_INDICES
    )
    return (
        ArmWaypoint(
            "boundary_lift_from_measured",
            tuple(final.tolist()),  # pitch 以外は実測を保つので値は使われない
            preserve_initial=preserve_except_pitch,
            offset_from_initial=tuple(
                (index, -lift) for index in _SHOULDER_PITCH_INDICES
            ),
        ),
        ArmWaypoint(
            "boundary_lifted_policy_initial_pose", tuple(lifted_final.tolist())
        ),
        ArmWaypoint("policy_initial_pose", tuple(final.tolist())),
    )


class GoLiveWaitSkill(Skill):
    """ロボットが指令に従い始めるまで待つ。頭の手順の先頭に置く (Issue #159 T5)。

    大会経路は client が繋がってから人が go-live (`wbc_driver.py --live`) を打つまで、
    運営 adapter が我々の指令をロボットに渡さない。その間も act は約 20 Hz で
    呼ばれるので、頭の手順がそのまま走ると手の watchdog が失敗し、腕の指令だけが
    経由点の先まで進んで、go-live の瞬間に一気に動く。運営から go-live を知らせる
    信号は無いので、「指令についてきたこと」を実測で見る。

    - 実測の腕から両肩 pitch を `lift_rad` だけ上げた姿勢 (負が上) を出し続ける。
      go-live で動くのはこの分だけ
    - どちらかの肩が `detect_rad` 以上上がったら完了。上げる向きは重力と逆なので、
      腕が垂れても誤検出しない
    - 時間の制限は無い (go-live が何分後でも待つ)
    - 検出できないときは、人が `flag_paths` のどれかに touch すれば強制できる。
      この skill が始まった後に作られたものだけ有効で、見たら消す
    - 自前経路は最初から従うので、数 tick で完了する
    """

    def __init__(
        self,
        *,
        name: str,
        lift_rad: float = 0.05,
        detect_rad: float = 0.02,
        flag_paths: Sequence[str] = (),
        reminder_interval_s: float = 10.0,
        time_fn: Callable[[], float] = time.monotonic,
        wall_time_fn: Callable[[], float] = time.time,
    ) -> None:
        super().__init__()
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        lift = float(lift_rad)
        detect = float(detect_rad)
        if not (np.isfinite(lift) and np.isfinite(detect) and 0.0 < detect < lift):
            raise ValueError(
                f"need 0 < detect_rad < lift_rad, got detect_rad={detect_rad} "
                f"lift_rad={lift_rad}"
            )
        self.name = name
        self._lift = lift
        self._detect = detect
        self._flag_paths = tuple(str(path) for path in flag_paths)
        self._reminder_interval_s = float(reminder_interval_s)
        self._time_fn = time_fn
        self._wall_time_fn = wall_time_fn
        self._reset()

    def _reset(self) -> None:
        self._origin: np.ndarray | None = None
        self._command: np.ndarray | None = None
        self._started_wall: float | None = None
        self._last_reminder_at: float | None = None
        self._complete = False
        self._completion_mode: str | None = None

    def _on_start(self, params: dict) -> None:
        self._reset()

    def _on_stop(self) -> None:
        pass

    def step(self, obs: dict) -> np.ndarray:
        measured = _measured_arm(obs)
        if self._origin is None:
            self._origin = measured.copy()
            self._command = measured.copy()
            for index in SHOULDER_PITCH_INDICES:
                self._command[index] -= self._lift
            self._command = np.clip(
                self._command, G1_ARM_POSITION_LOWER_RAD, G1_ARM_POSITION_UPPER_RAD
            )
            self._started_wall = self._wall_time_fn()
            self._last_reminder_at = self._time_fn()
            for path in self._flag_paths:  # 前の run の合図は消す
                self._remove(path)
            print(
                f"[go-live] {self.name}: waiting for the robot to follow "
                f"(both shoulders -{self._lift:g} rad, done at {self._detect:g} rad). "
                f"{self._force_hint()}",
                file=sys.stderr,
            )
            return self._command.copy()
        assert self._command is not None
        if self._complete:
            return self._command.copy()
        raised = max(
            float(self._origin[index] - measured[index])
            for index in SHOULDER_PITCH_INDICES
        )
        if raised >= self._detect:
            self._finish("followed", f"shoulders followed by {raised:.3f} rad")
        elif self._consume_flag():
            self._finish("forced", "forced by the flag file")
        else:
            now = self._time_fn()
            assert self._last_reminder_at is not None
            if now - self._last_reminder_at >= self._reminder_interval_s:
                self._last_reminder_at = now
                print(
                    f"[go-live] {self.name}: still waiting (followed {raised:.3f} / "
                    f"{self._detect:g} rad). {self._force_hint()}",
                    file=sys.stderr,
                )
        return self._command.copy()

    def _finish(self, mode: str, detail: str) -> None:
        self._complete = True
        self._completion_mode = mode
        print(f"[go-live] {self.name}: robot is live ({detail})", file=sys.stderr)

    def _force_hint(self) -> str:
        if not self._flag_paths:
            return ""
        return "To force: touch " + " or ".join(self._flag_paths)

    def _consume_flag(self) -> bool:
        assert self._started_wall is not None
        for path in self._flag_paths:
            try:
                modified = os.stat(path).st_mtime
            except OSError:
                continue
            if modified >= self._started_wall:
                self._remove(path)
                return True
        return False

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def failure_reason(self) -> str | None:
        return None

    @property
    def completion_mode(self) -> str | None:
        """`followed` (実測がついてきた) / `forced` (合図のファイル)。記録用。"""
        return self._completion_mode


class CollisionAwareArmPreMotionSkill(Skill):
    """Velocity/acceleration-limited waypoint follower with measured convergence."""

    name = "arm_pre_motion"

    def __init__(
        self,
        final_pose: Sequence[float],
        *,
        skill_name: str = "arm_pre_motion",
        velocity_limit_rad_s: float = 0.5,
        acceleration_limit_rad_s2: float = 1.0,
        measured_tolerance_rad: float = 0.10,
        measured_velocity_tolerance_rad_s: float = 0.05,
        command_tolerance_rad: float = 0.001,
        stage_timeout_s: float = 15.0,
        stage_wall_timeout_s: float | None = None,
        stable_samples_required: int = 5,
        published_target_provider: Callable[[], Sequence[float] | None] | None = None,
        measured_convergence_checker: Callable[
            [np.ndarray, np.ndarray, np.ndarray], tuple[bool, str]
        ]
        | None = None,
        time_fn: Callable[[], float] = time.monotonic,
        waypoint_profile: str = "overhead_clearance",
        boundary_lift_rad: float = 0.15,
    ) -> None:
        super().__init__()
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("skill_name must be a non-empty string")
        self.name = skill_name
        for label, value in (
            ("velocity_limit_rad_s", velocity_limit_rad_s),
            ("acceleration_limit_rad_s2", acceleration_limit_rad_s2),
            ("measured_tolerance_rad", measured_tolerance_rad),
            (
                "measured_velocity_tolerance_rad_s",
                measured_velocity_tolerance_rad_s,
            ),
            ("command_tolerance_rad", command_tolerance_rad),
            ("stage_timeout_s", stage_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive and finite")
        if (
            not isinstance(stable_samples_required, int)
            or isinstance(stable_samples_required, bool)
            or stable_samples_required <= 0
        ):
            raise ValueError("stable_samples_required must be a positive integer")
        # 経由点の選び方。歩行の後に腕を上げるときは頭上・側方の退避 (実績のある
        # 経路)、skill 間の境界は「少し上げて直接」(Issue #152)。
        if waypoint_profile == "overhead_clearance":
            self._waypoints = build_collision_aware_waypoints(final_pose)
        elif waypoint_profile == "boundary_lift":
            self._waypoints = build_boundary_lift_waypoints(
                final_pose, boundary_lift_rad
            )
        elif waypoint_profile == "reverse_clearance":
            self._waypoints = build_lowering_waypoints(final_pose)
        elif waypoint_profile == "direct":
            # 経由点なし。歩く前に腕を下ろすときのように、目標が既に安全な
            # 姿勢で、途中に回り道が要らない場合に使う (Issue #152)。
            final = np.asarray(final_pose, dtype=np.float64)
            if final.shape != (14,) or not np.isfinite(final).all():
                raise ValueError("final policy start pose must be finite 14-D")
            self._waypoints = (
                ArmWaypoint("policy_initial_pose", tuple(final.tolist())),
            )
        else:
            raise ValueError(
                "waypoint_profile must be 'overhead_clearance', 'boundary_lift', "
                f"'reverse_clearance' or 'direct', got {waypoint_profile!r}"
            )
        self._waypoint_profile = waypoint_profile
        self._velocity_limit = float(velocity_limit_rad_s)
        self._acceleration_limit = float(acceleration_limit_rad_s2)
        self._tolerance = float(measured_tolerance_rad)
        self._velocity_tolerance = float(measured_velocity_tolerance_rad_s)
        self._command_tolerance = float(command_tolerance_rad)
        self._stage_timeout = float(stage_timeout_s)
        # 実時間の上限 (Codex #12)。制御時間は 1 tick 0.1 s までしか進まないので、
        # tick が止まり気味だと実時間 60 s でも 6 s しか数えない。それとは別に
        # 「実時間でこれだけ経っても着かない」を失敗にする。model の読み込みで tick が
        # 数十秒止まる (実測 8〜27 s) のは正常なので、既定は制御時間の 4 倍と 60 s の
        # 大きい方。
        wall = (
            max(4.0 * self._stage_timeout, 60.0)
            if stage_wall_timeout_s is None
            else float(stage_wall_timeout_s)
        )
        if not math.isfinite(wall) or wall < self._stage_timeout:
            raise ValueError(
                "stage_wall_timeout_s must be finite and >= stage_timeout_s"
            )
        self._stage_wall_timeout = wall
        self._stable_samples_required = stable_samples_required
        self._published_target_provider = published_target_provider
        self._measured_convergence_checker = measured_convergence_checker
        self._time_fn = time_fn
        self._initial: np.ndarray | None = None
        self._targets: tuple[np.ndarray, ...] = ()
        self._command: np.ndarray | None = None
        self._velocity = np.zeros(14, dtype=np.float64)
        self._stage_index = 0
        self._stage_started_at: float | None = None
        # 締め切りは「実際に指令を進められた制御時間」で計る。tick thread が
        # 止まっている間 (model の読み込みで GIL を握られる / gc / camera の詰まり)
        # は実時間だけが進み、指令は 1 step 0.1 s 分しか進まない。実時間で計ると
        # 腕が正常に動いていても「着かなかった」と判定して遷移が落ちる (Issue #152)。
        self._stage_control_elapsed = 0.0
        self._last_at: float | None = None
        self._last_diagnostic_at: float | None = None
        self._stable_samples = 0
        self._complete = False
        self._failure_reason: str | None = None
        # 締め切りまでに目標へ届かなかった (故障ではない)。orchestrator は次へ進む。
        self._timeout_reason: str | None = None

    def _on_start(self, params: dict) -> None:
        self._initial = None
        self._targets = ()
        self._command = None
        self._velocity.fill(0.0)
        self._stage_index = 0
        self._stage_started_at = None
        self._stage_control_elapsed = 0.0
        self._last_at = None
        self._last_diagnostic_at = None
        self._stable_samples = 0
        self._complete = False
        self._failure_reason = None
        self._timeout_reason = None

    def _on_stop(self) -> None:
        pass

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def waypoint_profile(self) -> str:
        return self._waypoint_profile

    @property
    def waypoint_names(self) -> tuple[str, ...]:
        return tuple(waypoint.name for waypoint in self._waypoints)

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def timeout_reason(self) -> str | None:
        """締め切りまでに収束しなかった理由。orchestrator は時間切れとして次へ進む。"""
        return self._timeout_reason

    def step(self, obs: dict) -> np.ndarray:
        measured = _measured_arm(obs)
        measured_velocity = _measured_arm_velocity(obs)
        now = self._time_fn()
        if self._initial is None:
            self._initial = measured.copy()
            self._targets = tuple(w.resolve(self._initial) for w in self._waypoints)
            # A model boundary must begin from the target that actually reached
            # DDS, not snap the target buffer back to the lagging measured pose.
            # Startup paths have no provider and correctly bootstrap from the
            # measured pose.
            published = self._read_published_target()
            self._command = measured.copy() if published is None else published
            self._stage_started_at = now
            self._stage_control_elapsed = 0.0
            self._last_at = now
            self._last_diagnostic_at = now
            print(
                f"[pre-motion 1/{len(self._waypoints)}] "
                f"{self._waypoints[0].name} started",
                file=sys.stderr,
            )
            return self._command.copy()
        assert self._command is not None
        assert self._stage_started_at is not None
        assert self._last_at is not None
        if self._complete:
            return self._command.copy()
        if self._failure_reason is not None or self._timeout_reason is not None:
            return self._command.copy()
        # 積分と締め切りで同じ dt を使う (別々にすると上のずれが起きる)。
        dt = min(max(now - self._last_at, 1e-4), 0.1)
        self._stage_control_elapsed += dt
        wall_elapsed = now - self._stage_started_at
        if (
            self._stage_control_elapsed > self._stage_timeout
            or wall_elapsed > self._stage_wall_timeout
        ):
            waypoint = self._waypoints[self._stage_index]
            goal = self._targets[self._stage_index]
            errors = np.abs(goal - measured)
            worst = int(np.argmax(errors))
            error = float(errors[worst])
            self._timeout_reason = (
                f"pre-motion stage {waypoint.name!r} did not converge within "
                f"{self._stage_timeout:g}s control / {self._stage_wall_timeout:g}s "
                f"wall (control={self._stage_control_elapsed:.1f}s, "
                f"wall={wall_elapsed:.1f}s, max_arm_error={error:.4f}rad, "
                f"worst_joint={ARM_JOINT_NAMES[worst]}, "
                f"target={goal[worst]:+.4f}rad, measured={measured[worst]:+.4f}rad, "
                f"command={self._command[worst]:+.4f}rad)"
            )
            return self._command.copy()

        goal = self._targets[self._stage_index]
        error = goal - self._command
        braking_speed = np.sqrt(2.0 * self._acceleration_limit * np.abs(error))
        desired_velocity = np.sign(error) * np.minimum(
            self._velocity_limit, braking_speed
        )
        velocity_delta = np.clip(
            desired_velocity - self._velocity,
            -self._acceleration_limit * dt,
            self._acceleration_limit * dt,
        )
        self._velocity += velocity_delta
        step = self._velocity * dt
        overshoot = np.abs(step) >= np.abs(error)
        self._command += np.where(overshoot, error, step)
        self._velocity[overshoot] = 0.0
        self._last_at = now

        command_error = float(np.max(np.abs(goal - self._command)))
        published = self._read_published_target()
        published_error = (
            command_error
            if published is None
            else float(np.max(np.abs(goal - published)))
        )
        measured_error = float(np.max(np.abs(goal - measured)))
        measured_speed = float(np.max(np.abs(measured_velocity)))
        measurement_ok = (
            measured_error <= self._tolerance
            and measured_speed <= self._velocity_tolerance
        )
        convergence_detail = (
            f"joint_error={measured_error:.4f}rad speed={measured_speed:.3f}rad/s"
        )
        if self._measured_convergence_checker is not None:
            measurement_ok, convergence_detail = self._measured_convergence_checker(
                goal, measured, measured_velocity
            )
        within_tolerance = (
            command_error <= self._command_tolerance
            and published_error <= self._command_tolerance
            and measurement_ok
        )
        self._stable_samples = self._stable_samples + 1 if within_tolerance else 0
        assert self._last_diagnostic_at is not None
        if now - self._last_diagnostic_at >= 2.0 and not within_tolerance:
            errors = np.abs(goal - measured)
            worst = int(np.argmax(errors))
            print(
                f"[pre-motion {self._stage_index + 1}/{len(self._waypoints)}] "
                f"waiting: worst={ARM_JOINT_NAMES[worst]} "
                f"target={goal[worst]:+.3f} measured={measured[worst]:+.3f} "
                f"error={errors[worst]:.3f}rad {convergence_detail}",
                file=sys.stderr,
            )
            self._last_diagnostic_at = now
        if self._stable_samples >= self._stable_samples_required:
            waypoint = self._waypoints[self._stage_index]
            print(
                f"[pre-motion {self._stage_index + 1}/{len(self._waypoints)}] "
                f"{waypoint.name} reached ({convergence_detail})",
                file=sys.stderr,
            )
            self._stage_index += 1
            self._velocity.fill(0.0)
            self._stable_samples = 0
            self._stage_started_at = now
            self._stage_control_elapsed = 0.0
            self._last_diagnostic_at = now
            if self._stage_index >= len(self._waypoints):
                self._complete = True
            else:
                print(
                    f"[pre-motion {self._stage_index + 1}/{len(self._waypoints)}] "
                    f"{self._waypoints[self._stage_index].name} started",
                    file=sys.stderr,
                )
        return self._command.copy()

    def _read_published_target(self) -> np.ndarray | None:
        if self._published_target_provider is None:
            return None
        target = self._published_target_provider()
        if target is None:
            return None
        array = np.asarray(target, dtype=np.float64)
        if array.shape != (14,) or not np.isfinite(array).all():
            raise RuntimeError(
                "published arm target provider must return finite 14-D or None"
            )
        return array.copy()
