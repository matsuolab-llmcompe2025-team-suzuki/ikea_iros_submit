"""Post-walk collision-aware arm staging for the real Phase 1 profiles.

The robot walks while holding its measured, lowered Regular-Mode arm pose.
Only after the base has stopped do the arms follow the proven evaluation path:
shoulders backward, laterally outside the table, forward while still outside,
then the learned-policy start pose.  No waist, hand, or walking command is
issued by these skills.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from inference.desktop.lower_policy.actuators.base import WalkActuator
from inference.desktop.lower_policy.skills.base import Skill


ARM_INDICES = np.arange(15, 29, dtype=np.int64)
LEG_INDICES = np.arange(0, 12, dtype=np.int64)
SHOULDER_PITCH_INDICES = (0, 7)
SHOULDER_ROLL_INDICES = (1, 8)
ELBOW_INDICES = (3, 10)
LOWERED_WALK_TRACKING_MARGIN_RAD = 0.05
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


def _measured_max_leg_speed(obs: dict) -> float:
    state = obs.get("joint_state")
    velocities = None if state is None else getattr(state, "velocity", None)
    array = np.asarray(velocities, dtype=np.float64) if velocities is not None else None
    if array is None or array.shape != (29,) or not np.isfinite(array).all():
        raise RuntimeError(
            "live finite 29-D joint velocity is required before arm pre-motion"
        )
    return float(np.max(np.abs(array[LEG_INDICES])))


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
    checks = {
        "shoulder_pitch": (SHOULDER_PITCH_INDICES, 0.75),
        "shoulder_roll": (SHOULDER_ROLL_INDICES, 0.65),
        "elbow": (ELBOW_INDICES, 1.00),
    }
    for label, (indices, maximum) in checks.items():
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
    """Hold a verified lowered measured pose before and throughout walking."""

    name = "setup"

    def __init__(self, *, dwell_sec: float = 0.5) -> None:
        super().__init__()
        if not math.isfinite(dwell_sec) or dwell_sec <= 0.0:
            raise ValueError("dwell_sec must be positive and finite")
        self._dwell_sec = float(dwell_sec)
        self._hold: np.ndarray | None = None

    def _on_start(self, params: dict) -> None:
        self._hold = None

    def _on_stop(self) -> None:
        pass

    def step(self, obs: dict) -> np.ndarray:
        if self._hold is None:
            self._hold = validate_lowered_walk_pose(_measured_arm(obs))
            print(
                "[setup] measured lowered arm pose latched; walking will keep this pose",
                file=sys.stderr,
            )
        return self._hold.copy()

    @property
    def max_dwell_sec(self) -> float:
        return self._dwell_sec


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

    def resolve(self, initial: np.ndarray) -> np.ndarray:
        target = np.asarray(self.target, dtype=np.float64).copy()
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError(f"waypoint {self.name!r} must be finite 14-D")
        for index in self.preserve_initial:
            target[index] = initial[index]
        for index in self.do_not_decrease_from_initial:
            target[index] = max(target[index], initial[index])
        if np.max(np.abs(target)) > 1.5:
            raise ValueError(f"waypoint {self.name!r} exceeds the 1.5rad smoke limit")
        return target


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
        stage_timeout_s: float = 15.0,
        stable_samples_required: int = 5,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("skill_name must be a non-empty string")
        self.name = skill_name
        for label, value in (
            ("velocity_limit_rad_s", velocity_limit_rad_s),
            ("acceleration_limit_rad_s2", acceleration_limit_rad_s2),
            ("measured_tolerance_rad", measured_tolerance_rad),
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
        self._waypoints = build_collision_aware_waypoints(final_pose)
        self._velocity_limit = float(velocity_limit_rad_s)
        self._acceleration_limit = float(acceleration_limit_rad_s2)
        self._tolerance = float(measured_tolerance_rad)
        self._stage_timeout = float(stage_timeout_s)
        self._stable_samples_required = stable_samples_required
        self._time_fn = time_fn
        self._initial: np.ndarray | None = None
        self._targets: tuple[np.ndarray, ...] = ()
        self._command: np.ndarray | None = None
        self._velocity = np.zeros(14, dtype=np.float64)
        self._stage_index = 0
        self._stage_started_at: float | None = None
        self._last_at: float | None = None
        self._last_diagnostic_at: float | None = None
        self._stable_samples = 0
        self._complete = False

    def _on_start(self, params: dict) -> None:
        self._initial = None
        self._targets = ()
        self._command = None
        self._velocity.fill(0.0)
        self._stage_index = 0
        self._stage_started_at = None
        self._last_at = None
        self._last_diagnostic_at = None
        self._stable_samples = 0
        self._complete = False

    def _on_stop(self) -> None:
        pass

    @property
    def is_complete(self) -> bool:
        return self._complete

    def step(self, obs: dict) -> np.ndarray:
        measured = _measured_arm(obs)
        now = self._time_fn()
        if self._initial is None:
            self._initial = measured.copy()
            self._targets = tuple(w.resolve(self._initial) for w in self._waypoints)
            self._command = measured.copy()
            self._stage_started_at = now
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
        if now - self._stage_started_at > self._stage_timeout:
            waypoint = self._waypoints[self._stage_index]
            goal = self._targets[self._stage_index]
            errors = np.abs(goal - measured)
            worst = int(np.argmax(errors))
            error = float(errors[worst])
            raise TimeoutError(
                f"pre-motion stage {waypoint.name!r} did not converge within "
                f"{self._stage_timeout:g}s (max_arm_error={error:.4f}rad, "
                f"worst_joint={ARM_JOINT_NAMES[worst]}, "
                f"target={goal[worst]:+.4f}rad, measured={measured[worst]:+.4f}rad, "
                f"command={self._command[worst]:+.4f}rad)"
            )

        dt = min(max(now - self._last_at, 1e-4), 0.1)
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
        measured_error = float(np.max(np.abs(goal - measured)))
        within_tolerance = command_error <= 1e-3 and measured_error <= self._tolerance
        self._stable_samples = self._stable_samples + 1 if within_tolerance else 0
        assert self._last_diagnostic_at is not None
        if now - self._last_diagnostic_at >= 2.0 and not within_tolerance:
            errors = np.abs(goal - measured)
            worst = int(np.argmax(errors))
            print(
                f"[pre-motion {self._stage_index + 1}/{len(self._waypoints)}] "
                f"waiting: worst={ARM_JOINT_NAMES[worst]} "
                f"target={goal[worst]:+.3f} measured={measured[worst]:+.3f} "
                f"error={errors[worst]:.3f}rad",
                file=sys.stderr,
            )
            self._last_diagnostic_at = now
        if self._stable_samples >= self._stable_samples_required:
            waypoint = self._waypoints[self._stage_index]
            print(
                f"[pre-motion {self._stage_index + 1}/{len(self._waypoints)}] "
                f"{waypoint.name} reached (error={measured_error:.4f}rad)",
                file=sys.stderr,
            )
            self._stage_index += 1
            self._velocity.fill(0.0)
            self._stable_samples = 0
            self._stage_started_at = now
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
