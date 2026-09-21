"""Deterministic post-carry handover for the pick-leg hybrid controller.

Phase 2 ends with the right hand holding the leg and the open left hand at the
first handover pose.  The original hybrid then returned control to GR00T.  This
module provides an opt-in finite-state alternative which completes the
canonical dataset sequence:

    left grasp -> right release -> left presents -> right regrasp ->
    left release -> left arm to insert frame-0 -> right arm to insert frame-0
    -> both hands to insert frame-0

Only one arm moves in any stage.  In particular, an arm never moves while both
hands are intentionally holding the same rigid leg.  Grasp transitions require
measured Dex1 obstruction, and release transitions require measured opening.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from inference.desktop.pick_leg_hybrid.interlock import (
    LEFT,
    RIGHT,
    GraspInterlockConfig,
    is_grasping,
)
from inference.desktop.pick_leg_hybrid.phase2 import (
    EePose,
    _lerp_pose,
    quat_angle_between,
)


class Phase3RuleStage(str, Enum):
    CLOSE_LEFT = "close_left"
    OPEN_RIGHT = "open_right"
    MOVE_LEFT_PRESENT = "move_left_present"
    MOVE_RIGHT_REGRASP = "move_right_regrasp"
    CLOSE_RIGHT = "close_right"
    OPEN_LEFT = "open_left"
    MOVE_LEFT_INSERT_INITIAL = "move_left_insert_initial"
    MOVE_RIGHT_INSERT_INITIAL = "move_right_insert_initial"
    SET_INSERT_HAND_INITIAL = "set_insert_hand_initial"
    COMPLETE = "complete"


_ORDER = tuple(Phase3RuleStage)
_MOVING_SIDE: dict[Phase3RuleStage, int | None] = {
    Phase3RuleStage.CLOSE_LEFT: None,
    Phase3RuleStage.OPEN_RIGHT: None,
    Phase3RuleStage.MOVE_LEFT_PRESENT: LEFT,
    Phase3RuleStage.MOVE_RIGHT_REGRASP: RIGHT,
    Phase3RuleStage.CLOSE_RIGHT: None,
    Phase3RuleStage.OPEN_LEFT: None,
    Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL: LEFT,
    Phase3RuleStage.MOVE_RIGHT_INSERT_INITIAL: RIGHT,
    Phase3RuleStage.SET_INSERT_HAND_INITIAL: None,
    Phase3RuleStage.COMPLETE: None,
}


@dataclass(frozen=True)
class RuleStageTiming:
    duration_sec: float
    timeout_sec: float

    def __post_init__(self) -> None:
        if self.duration_sec <= 0.0:
            raise ValueError("phase3 rule duration_sec must be > 0")
        if self.timeout_sec < self.duration_sec:
            raise ValueError("phase3 rule timeout_sec must be >= duration_sec")


@dataclass(frozen=True)
class Phase3RuleConfig:
    left_present: EePose
    right_regrasp_offset_left_local: np.ndarray
    right_regrasp_quat: np.ndarray
    close_hand_value: float
    open_hand_value: float
    right_initial_hold_value: float
    grasp_preload_rad: float
    open_min: float
    confirm_count: int
    pos_tol: float
    rot_tol: float
    left_present_pos_tol: float
    left_present_rot_tol: float
    joint_tol: float
    joint_velocity_tol: float
    joint_command_tol: float
    hand_tol: float
    timings: Mapping[Phase3RuleStage, RuleStageTiming]

    def __post_init__(self) -> None:
        offset = np.asarray(self.right_regrasp_offset_left_local, dtype=np.float64)
        quat = np.asarray(self.right_regrasp_quat, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("phase3 right regrasp offset must be finite (3,)")
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            raise ValueError("phase3 right regrasp quaternion must be finite (4,)")
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            raise ValueError("phase3 right regrasp quaternion must be non-zero")
        object.__setattr__(self, "right_regrasp_offset_left_local", offset)
        object.__setattr__(self, "right_regrasp_quat", quat / norm)
        if not self.close_hand_value < self.open_hand_value:
            raise ValueError("phase3 close_hand_value must be below open_hand_value")
        if not np.isfinite(self.grasp_preload_rad) or self.grasp_preload_rad <= 0.0:
            raise ValueError("phase3 grasp_preload_rad must be positive and finite")
        if self.confirm_count < 1:
            raise ValueError("phase3 confirm_count must be >= 1")
        if min(
            self.pos_tol,
            self.rot_tol,
            self.left_present_pos_tol,
            self.left_present_rot_tol,
            self.joint_tol,
            self.joint_velocity_tol,
            self.joint_command_tol,
            self.hand_tol,
        ) <= 0.0:
            raise ValueError("phase3 tolerances must be > 0")
        required = set(_ORDER) - {Phase3RuleStage.COMPLETE}
        if set(self.timings) != required:
            raise ValueError(
                "phase3 timings must define exactly "
                f"{sorted(stage.value for stage in required)}"
            )


@dataclass(frozen=True)
class Phase3RuleCommand:
    left: EePose
    right: EePose
    hand: np.ndarray
    stage: Phase3RuleStage
    moving_side: int | None
    progress: float
    complete: bool
    failure_reason: str | None
    arm_target: np.ndarray | None


class Phase3RuleController:
    """Observation-driven post-carry FSM with fail-closed timeouts."""

    def __init__(
        self,
        cfg: Phase3RuleConfig,
        grasp_cfg: GraspInterlockConfig,
        insert_arm_target: np.ndarray,
        insert_hand_target: np.ndarray,
    ) -> None:
        self.cfg = cfg
        self._grasp_cfg = grasp_cfg
        self._insert_arm_target = _finite_vector(
            insert_arm_target, 14, "insert arm target"
        )
        self._insert_hand_target = _finite_vector(
            insert_hand_target, 2, "insert hand target"
        )
        if np.any(self._insert_hand_target < cfg.close_hand_value) or np.any(
            self._insert_hand_target > cfg.open_hand_value
        ):
            raise ValueError(
                "insert hand target must lie between phase3 close/open commands, "
                f"got {self._insert_hand_target}"
            )
        if cfg.grasp_preload_rad <= grasp_cfg.follow_error_min:
            raise ValueError(
                "phase3 grasp_preload_rad must exceed the grasp interlock "
                f"follow_error_min ({grasp_cfg.follow_error_min})"
            )
        self.reset()

    @property
    def stage(self) -> Phase3RuleStage:
        return self._stage

    @property
    def complete(self) -> bool:
        return self._stage is Phase3RuleStage.COMPLETE

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    def reset(self) -> None:
        self._stage = Phase3RuleStage.CLOSE_LEFT
        self._started = False
        self._stage_t0 = 0.0
        self._start_left: EePose | None = None
        self._start_right: EePose | None = None
        self._goal_left: EePose | None = None
        self._goal_right: EePose | None = None
        self._start_arm: np.ndarray | None = None
        self._confirm_hits = 0
        self._hold_loss_hits = 0
        self._failure_reason: str | None = None
        self._left_grasp_hold: float | None = None
        self._right_grasp_hold: float | None = None

    def start(
        self, t: float, left: EePose, right: EePose, arm_state: np.ndarray
    ) -> None:
        self.reset()
        self._started = True
        self._enter(self._stage, float(t), left, right, arm_state)

    def step(
        self,
        *,
        t: float,
        left: EePose,
        right: EePose,
        hand_state: np.ndarray,
        previous_hand_command: np.ndarray,
        arm_state: np.ndarray,
        arm_velocity: np.ndarray | None = None,
        previous_arm_command: np.ndarray | None = None,
    ) -> Phase3RuleCommand:
        if not self._started:
            raise RuntimeError("Phase3RuleController.start() must be called first")
        if self._failure_reason is not None or self.complete:
            return self._command(float(t))

        now = float(t)
        self._check_supporting_grasp(hand_state, previous_hand_command)
        if self._failure_reason is not None:
            return self._command(now)
        timing = self.cfg.timings[self._stage]
        elapsed = now - self._stage_t0
        arm = _finite_vector(arm_state, 14, "phase3 arm state")
        velocity = (
            np.zeros(14, dtype=np.float64)
            if arm_velocity is None
            else _finite_vector(arm_velocity, 14, "phase3 arm velocity")
        )
        previous_arm = (
            arm.copy()
            if previous_arm_command is None
            else _finite_vector(
                previous_arm_command, 14, "phase3 previous arm command"
            )
        )
        if self._satisfied(
            left,
            right,
            hand_state,
            previous_hand_command,
            arm,
            velocity,
            previous_arm,
            elapsed,
        ):
            self._capture_grasp_hold(hand_state)
            self._advance(now, left, right, arm)
        elif elapsed > timing.timeout_sec:
            detail = self._timeout_detail(
                left,
                right,
                hand_state,
                previous_hand_command,
                arm,
                velocity,
                previous_arm,
            )
            self._failure_reason = (
                f"phase3 rule stage {self._stage.value!r} timed out after "
                f"{elapsed:.2f}s{detail}"
            )
        return self._command(now)

    def _capture_grasp_hold(self, hand_state: np.ndarray) -> None:
        """Replace the full-close probe with a small measured preload.

        ``close_hand_value`` is only a contact probe.  Continuing to command it
        after the object has stopped the gripper leaves a multi-radian position
        error and can latch the Dex1 winding-overtemperature fault.  Once the
        obstruction is confirmed, retain just enough error for the same
        state-command interlock to remain observable.
        """
        state = np.asarray(hand_state, dtype=np.float64)
        if self._stage is Phase3RuleStage.CLOSE_LEFT:
            self._left_grasp_hold = float(
                np.clip(
                    state[LEFT] - self.cfg.grasp_preload_rad,
                    self.cfg.close_hand_value,
                    self.cfg.open_hand_value,
                )
            )
        elif self._stage is Phase3RuleStage.CLOSE_RIGHT:
            self._right_grasp_hold = float(
                np.clip(
                    state[RIGHT] - self.cfg.grasp_preload_rad,
                    self.cfg.close_hand_value,
                    self.cfg.open_hand_value,
                )
            )

    def _timeout_detail(
        self,
        left: EePose,
        right: EePose,
        hand_state: np.ndarray,
        previous_hand_command: np.ndarray,
        arm_state: np.ndarray,
        arm_velocity: np.ndarray,
        previous_arm_command: np.ndarray,
    ) -> str:
        """Return actionable measured-error diagnostics for a failed stage."""
        if self._stage in {
            Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL,
            Phase3RuleStage.MOVE_RIGHT_INSERT_INITIAL,
        }:
            side = LEFT if self._stage is Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL else RIGHT
            arm_slice = slice(0, 7) if side == LEFT else slice(7, 14)
            errors = np.abs(
                arm_state[arm_slice] - self._insert_arm_target[arm_slice]
            )
            worst = int(np.argmax(errors))
            label = "left" if side == LEFT else "right"
            velocity = float(np.max(np.abs(arm_velocity[arm_slice])))
            command_error = float(
                np.max(
                    np.abs(
                        previous_arm_command[arm_slice]
                        - self._insert_arm_target[arm_slice]
                    )
                )
            )
            return (
                f" ({label}_arm_max_error={float(errors[worst]):.4f}rad, "
                f"joint_index={worst}, tolerance={self.cfg.joint_tol:.4f}rad, "
                f"max_velocity={velocity:.4f}rad/s, "
                f"velocity_tolerance={self.cfg.joint_velocity_tol:.4f}rad/s, "
                f"command_error={command_error:.4f}rad, "
                f"command_tolerance={self.cfg.joint_command_tol:.4f}rad)"
            )
        if self._stage is Phase3RuleStage.SET_INSERT_HAND_INITIAL:
            errors = np.abs(
                np.asarray(hand_state, dtype=np.float64) - self._insert_hand_target
            )
            return (
                f" (hand_error={errors.tolist()}, "
                f"tolerance={self.cfg.hand_tol:.4f}rad)"
            )
        if self._stage in {
            Phase3RuleStage.MOVE_LEFT_PRESENT,
            Phase3RuleStage.MOVE_RIGHT_REGRASP,
        }:
            actual = left if self._stage is Phase3RuleStage.MOVE_LEFT_PRESENT else right
            goal = self._goal_left if self._stage is Phase3RuleStage.MOVE_LEFT_PRESENT else self._goal_right
            if goal is not None:
                pos_tol = (
                    self.cfg.left_present_pos_tol
                    if self._stage is Phase3RuleStage.MOVE_LEFT_PRESENT
                    else self.cfg.pos_tol
                )
                rot_tol = (
                    self.cfg.left_present_rot_tol
                    if self._stage is Phase3RuleStage.MOVE_LEFT_PRESENT
                    else self.cfg.rot_tol
                )
                return (
                    f" (position_error={float(np.linalg.norm(actual.pos - goal.pos)):.4f}m, "
                    f"position_tolerance={pos_tol:.4f}m, "
                    f"rotation_error={quat_angle_between(actual.quat, goal.quat):.4f}rad, "
                    f"rotation_tolerance={rot_tol:.4f}rad)"
                )
        return ""

    def _check_supporting_grasp(
        self, hand_state: np.ndarray, previous_hand_command: np.ndarray
    ) -> None:
        """Latch failure if the only supporting hand loses the leg."""
        support_side: int | None = None
        if self._stage in {
            Phase3RuleStage.OPEN_RIGHT,
            Phase3RuleStage.MOVE_LEFT_PRESENT,
            Phase3RuleStage.MOVE_RIGHT_REGRASP,
            Phase3RuleStage.CLOSE_RIGHT,
        }:
            support_side = LEFT
        elif self._stage in {
            Phase3RuleStage.OPEN_LEFT,
            Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL,
            Phase3RuleStage.MOVE_RIGHT_INSERT_INITIAL,
            Phase3RuleStage.SET_INSERT_HAND_INITIAL,
            Phase3RuleStage.COMPLETE,
        }:
            support_side = RIGHT
        if support_side is None:
            self._hold_loss_hits = 0
            return
        holding = is_grasping(
            hand_state, previous_hand_command, support_side, self._grasp_cfg
        )
        self._hold_loss_hits = 0 if holding else self._hold_loss_hits + 1
        if self._hold_loss_hits >= self.cfg.confirm_count:
            label = "left" if support_side == LEFT else "right"
            self._failure_reason = (
                f"phase3 rule lost the {label} supporting grasp during "
                f"{self._stage.value!r}"
            )

    def _satisfied(
        self,
        left: EePose,
        right: EePose,
        hand_state: np.ndarray,
        previous_hand_command: np.ndarray,
        arm_state: np.ndarray,
        arm_velocity: np.ndarray,
        previous_arm_command: np.ndarray,
        elapsed: float,
    ) -> bool:
        state = np.asarray(hand_state, dtype=np.float64)
        command = np.asarray(previous_hand_command, dtype=np.float64)
        if state.shape != (2,) or command.shape != (2,):
            raise ValueError("phase3 rule requires bilateral Dex1 state and command")

        condition = False
        if self._stage is Phase3RuleStage.CLOSE_LEFT:
            condition = is_grasping(state, command, LEFT, self._grasp_cfg)
        elif self._stage is Phase3RuleStage.OPEN_RIGHT:
            condition = bool(
                state[RIGHT] >= self.cfg.open_min
                and command[RIGHT] >= self.cfg.open_min
            )
        elif self._stage is Phase3RuleStage.MOVE_LEFT_PRESENT:
            condition = self._pose_reached(
                left,
                self._goal_left,
                pos_tol=self.cfg.left_present_pos_tol,
                rot_tol=self.cfg.left_present_rot_tol,
            )
        elif self._stage is Phase3RuleStage.MOVE_RIGHT_REGRASP:
            condition = self._pose_reached(right, self._goal_right)
        elif self._stage is Phase3RuleStage.CLOSE_RIGHT:
            condition = is_grasping(state, command, RIGHT, self._grasp_cfg)
        elif self._stage is Phase3RuleStage.OPEN_LEFT:
            condition = bool(
                state[LEFT] >= self.cfg.open_min
                and command[LEFT] >= self.cfg.open_min
            )
        elif self._stage is Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL:
            condition = self._insert_arm_reached(
                LEFT, arm_state, arm_velocity, previous_arm_command, elapsed
            )
        elif self._stage is Phase3RuleStage.MOVE_RIGHT_INSERT_INITIAL:
            condition = self._insert_arm_reached(
                RIGHT, arm_state, arm_velocity, previous_arm_command, elapsed
            )
        elif self._stage is Phase3RuleStage.SET_INSERT_HAND_INITIAL:
            left_ready = bool(
                abs(state[LEFT] - self._insert_hand_target[LEFT])
                <= self.cfg.hand_tol
            )
            right_holding = is_grasping(
                state, command, RIGHT, self._grasp_cfg
            )
            condition = left_ready and right_holding

        self._confirm_hits = self._confirm_hits + 1 if condition else 0
        return self._confirm_hits >= self.cfg.confirm_count

    def _insert_arm_reached(
        self,
        side: int,
        arm_state: np.ndarray,
        arm_velocity: np.ndarray,
        previous_arm_command: np.ndarray,
        elapsed: float,
    ) -> bool:
        """Confirm a settled, fully transmitted insert-frame arm target.

        Physical arm_sdk tracking has a small gravity/compliance residue even
        after the command has converged.  Position tolerance alone either
        strands the FSM on that residue or, if widened without context, can
        accept a still-moving/partially-transmitted command.  Require all
        three signals so only a stationary final command can advance.
        """
        arm_slice = slice(0, 7) if side == LEFT else slice(7, 14)
        target = self._insert_arm_target[arm_slice]
        timing = self.cfg.timings[self._stage]
        return bool(
            elapsed >= timing.duration_sec
            and np.max(np.abs(arm_state[arm_slice] - target)) <= self.cfg.joint_tol
            and np.max(np.abs(arm_velocity[arm_slice]))
            <= self.cfg.joint_velocity_tol
            and np.max(np.abs(previous_arm_command[arm_slice] - target))
            <= self.cfg.joint_command_tol
        )

    def _pose_reached(
        self,
        actual: EePose,
        goal: EePose | None,
        *,
        pos_tol: float | None = None,
        rot_tol: float | None = None,
    ) -> bool:
        if goal is None:
            return False
        position_tolerance = self.cfg.pos_tol if pos_tol is None else float(pos_tol)
        rotation_tolerance = self.cfg.rot_tol if rot_tol is None else float(rot_tol)
        return bool(
            np.linalg.norm(actual.pos - goal.pos) <= position_tolerance
            and quat_angle_between(actual.quat, goal.quat) <= rotation_tolerance
        )

    def _advance(
        self, t: float, left: EePose, right: EePose, arm_state: np.ndarray
    ) -> None:
        index = _ORDER.index(self._stage)
        self._stage = _ORDER[index + 1]
        self._confirm_hits = 0
        self._hold_loss_hits = 0
        if not self.complete:
            self._enter(self._stage, t, left, right, arm_state)

    def _enter(
        self,
        stage: Phase3RuleStage,
        t: float,
        left: EePose,
        right: EePose,
        arm_state: np.ndarray,
    ) -> None:
        self._stage_t0 = float(t)
        self._start_left = left
        self._start_right = right
        self._goal_left = left
        self._goal_right = right
        self._start_arm = _finite_vector(
            arm_state, 14, "phase3 stage-entry arm state"
        ).copy()
        if stage is Phase3RuleStage.MOVE_LEFT_PRESENT:
            self._goal_left = self.cfg.left_present
        elif stage is Phase3RuleStage.MOVE_RIGHT_REGRASP:
            left_rotation = _quat_to_matrix(left.quat)
            position = left.pos + left_rotation @ self.cfg.right_regrasp_offset_left_local
            self._goal_right = EePose.of(position, self.cfg.right_regrasp_quat)

    def _command(self, t: float) -> Phase3RuleCommand:
        assert self._start_left is not None and self._start_right is not None
        left = self._start_left
        right = self._start_right
        progress = 1.0 if self.complete else self._progress(t)
        if self._stage is Phase3RuleStage.MOVE_LEFT_PRESENT:
            assert self._goal_left is not None
            left = _lerp_pose(self._start_left, self._goal_left, progress)
        elif self._stage is Phase3RuleStage.MOVE_RIGHT_REGRASP:
            assert self._goal_right is not None
            right = _lerp_pose(self._start_right, self._goal_right, progress)
        arm_target: np.ndarray | None = None
        if self._stage in {
            Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL,
            Phase3RuleStage.MOVE_RIGHT_INSERT_INITIAL,
        }:
            assert self._start_arm is not None
            arm_target = self._start_arm.copy()
            arm_slice = (
                slice(0, 7)
                if self._stage is Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL
                else slice(7, 14)
            )
            # This command is subsequently passed through the shared
            # acceleration/braking-aware MotionLimiter.  Interpolating here as
            # well creates two cascaded trajectory generators: the outer
            # limiter continually chases a moving intermediate goal and can
            # still lag the final target when this stage's timeout expires.
            # Supply the final one-arm goal immediately and let the single
            # safety limiter generate the bounded trajectory.  The other arm
            # remains exactly at its measured stage-entry pose.
            arm_target[arm_slice] = self._insert_arm_target[arm_slice]

        hand = np.array([
            (
                self.cfg.close_hand_value
                if self._left_grasp_hold is None
                else self._left_grasp_hold
            ),
            (
                self.cfg.right_initial_hold_value
                if self._right_grasp_hold is None
                else self._right_grasp_hold
            ),
        ], dtype=np.float64)
        if self._stage in {
            Phase3RuleStage.OPEN_RIGHT,
            Phase3RuleStage.MOVE_LEFT_PRESENT,
            Phase3RuleStage.MOVE_RIGHT_REGRASP,
        }:
            hand[RIGHT] = self.cfg.open_hand_value
        elif self._stage in {
            Phase3RuleStage.CLOSE_RIGHT,
            Phase3RuleStage.OPEN_LEFT,
            Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL,
            Phase3RuleStage.MOVE_RIGHT_INSERT_INITIAL,
            Phase3RuleStage.SET_INSERT_HAND_INITIAL,
            Phase3RuleStage.COMPLETE,
        }:
            hand[RIGHT] = (
                self.cfg.close_hand_value
                if self._right_grasp_hold is None
                else self._right_grasp_hold
            )
        if self._stage in {
            Phase3RuleStage.OPEN_LEFT,
            Phase3RuleStage.MOVE_LEFT_INSERT_INITIAL,
            Phase3RuleStage.MOVE_RIGHT_INSERT_INITIAL,
            Phase3RuleStage.COMPLETE,
        }:
            hand[LEFT] = self.cfg.open_hand_value
        if self._stage in {
            Phase3RuleStage.SET_INSERT_HAND_INITIAL,
            Phase3RuleStage.COMPLETE,
        }:
            hand = self._insert_hand_target.copy()

        return Phase3RuleCommand(
            left=left,
            right=right,
            hand=hand,
            stage=self._stage,
            moving_side=_MOVING_SIDE[self._stage],
            progress=progress,
            complete=self.complete,
            failure_reason=self._failure_reason,
            arm_target=arm_target,
        )

    def _progress(self, t: float) -> float:
        timing = self.cfg.timings[self._stage]
        return float(min(1.0, max(0.0, (float(t) - self._stage_t0) / timing.duration_sec)))


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ],
        dtype=np.float64,
    )


def _finite_vector(value: np.ndarray, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite ({size},), got {array}")
    return array
