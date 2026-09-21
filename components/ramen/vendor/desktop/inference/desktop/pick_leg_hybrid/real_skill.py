"""Real-G1 adapter for the Issue #136 VLM/VLA/MP pick-leg controller.

The original Issue #136 controller emits the competition boundary's 25-D
task-space action and assumes an external IK service.  The physical runner in
this repository accepts absolute arm joint targets instead.  This adapter
therefore keeps the same phase logic, but converts phase-2 Cartesian waypoints
to 14 arm joints with the repository's tested G1 IK and then applies the same
motion limiter used by every other real VLA skill.

It never sends waist or leg targets.  VLM failure is fail-closed: phase 1 keeps
running and cannot transition to phase 2.
"""

from __future__ import annotations

import base64
import json
import socket
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from inference.desktop.lower_policy.kinematics.g1_arm import (
    DEX1_BASE_OFFSET,
    G1ArmKinematics,
)
from inference.desktop.lower_policy.kinematics.types import EEPose, Side, matrix_to_rpy
from inference.desktop.lower_policy.policies.base import PolicyAction
from inference.desktop.lower_policy.skills.vla_skill import (
    ACTION_DIM_TOTAL,
    ARMS_SLICE,
    HAND_SLICE,
    WAIST_SLICE,
    PickTableLegVlaSkill,
    split_head_stereo,
)
from inference.desktop.perception.g1_urdf_fk import (
    LEFT_WRIST_TOOL_OFFSET_M,
    RIGHT_WRIST_TOOL_OFFSET_M,
)
from inference.desktop.pick_leg_hybrid.boundary import GraspBoundaryDetector
from inference.desktop.pick_leg_hybrid.config import (
    DEFAULT_CONFIG_PATH,
    PickLegHybridConfig,
    load_yaml,
)
from inference.desktop.pick_leg_hybrid.interlock import RIGHT, is_grasping
from inference.desktop.pick_leg_hybrid.overlay_input import FrameHistory
from inference.desktop.pick_leg_hybrid.phase2 import EePose, Phase2Motion
from inference.desktop.pick_leg_hybrid.phase3_rule import Phase3RuleController
from inference.desktop.pick_leg_hybrid.phases import Phase, PhaseStateMachine
from inference.desktop.pick_leg_hybrid.vlm import GRASP_QUESTION, VlmBoundaryClient


def _resolve_asset(config_path: Path, value: str | None) -> Path:
    if not value:
        raise ValueError("pick-leg hybrid requires both VLM reference images")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pick-leg VLM reference image not found: {path}")
    return path


def load_reference_images(
    config_path: str | Path,
    *,
    endpoint_override: str | None = None,
    model_override: str | None = None,
) -> tuple[PickLegHybridConfig, list[str]]:
    """Load the config and two pinned JPEG references, failing closed."""
    path = Path(config_path).expanduser().resolve()
    cfg = load_yaml(path)
    if endpoint_override is not None or model_override is not None:
        cfg = replace(
            cfg,
            vlm=replace(
                cfg.vlm,
                endpoint=(endpoint_override or cfg.vlm.endpoint),
                model=(model_override or cfg.vlm.model),
            ),
        )
    before = _resolve_asset(path, cfg.references.before)
    after = _resolve_asset(path, cfg.references.after)
    encoded = [
        base64.b64encode(asset.read_bytes()).decode("ascii")
        for asset in (before, after)
    ]
    return cfg, encoded


def probe_vlm_endpoint(
    cfg: PickLegHybridConfig,
    reference_images_b64: list[str] | None = None,
    timeout_sec: float = 3.0,
) -> dict:
    """Read-only endpoint check used before any actuator is constructed."""
    parsed = urllib.parse.urlparse(cfg.vlm.endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"invalid VLM endpoint: {cfg.vlm.endpoint!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout_sec):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"VLM endpoint is not reachable at {parsed.hostname}:{port}: {exc}"
        ) from exc
    root = f"{parsed.scheme}://{parsed.netloc}"
    request = urllib.request.Request(root + "/v1/models", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"VLM /v1/models preflight failed at {root}: {exc}") from exc
    ids = {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    if ids and cfg.vlm.model not in ids:
        raise RuntimeError(
            f"VLM model {cfg.vlm.model!r} is not served by {root}; available={sorted(ids)}"
        )
    result = {
        "endpoint": cfg.vlm.endpoint,
        "model": cfg.vlm.model,
        "served_models": sorted(ids),
    }
    if reference_images_b64:
        # Exercise the *same image cardinality as the runtime request*.
        # Runtime prepends reference A/B to ``history_frames`` old frames plus
        # the current frame.  A smaller preflight previously passed against a
        # vLLM server limited to four images while every five-image runtime
        # request failed with HTTP 400.
        self_check_images = build_vlm_self_check_images(cfg, reference_images_b64)
        # Use B for every live/history slot: the expected answer is
        # unambiguously 1 while still validating the production request shape.
        answer = VlmBoundaryClient(cfg.vlm).ask(
            GRASP_QUESTION,
            self_check_images,
            current_phase_text="右手でテーブル脚へ接近し把持する",
        )
        if answer.value != 1:
            raise RuntimeError(
                "VLM multimodal self-check failed: reference B as current frame "
                f"must produce answer=1, got value={answer.value!r} "
                f"error={answer.error!r} raw={answer.raw[:160]!r}"
            )
        result.update({
            "multimodal_self_check": "passed",
            "multimodal_latency_sec": float(answer.latency_sec),
            "runtime_image_count": len(self_check_images),
        })
    return result


def build_vlm_self_check_images(
    cfg: PickLegHybridConfig, reference_images_b64: list[str]
) -> list[str]:
    """Return a deterministic request with production runtime cardinality."""

    if len(reference_images_b64) != 2:
        raise ValueError(
            "pick-leg VLM self-check requires exactly two reference images"
        )
    runtime_frames = [reference_images_b64[1]] * (cfg.vlm.history_frames + 1)
    return [*reference_images_b64, *runtime_frames]


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k])) * 2.0
        q = np.zeros(4)
        q[i + 1] = 0.25 * s
        q[0] = (m[k, j] - m[j, k]) / s
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


class RealPickLegHybridVlaSkill(PickTableLegVlaSkill):
    """VLA -> VLM boundary -> joint-space MP -> VLA or rule-based Phase 3."""

    def __init__(
        self,
        *args: Any,
        hybrid_config_path: str | Path = DEFAULT_CONFIG_PATH,
        hybrid_vlm_endpoint: str | None = None,
        hybrid_vlm_model: str | None = None,
        phase3_executor: str = "vla",
        next_initial_arm_target: np.ndarray | None = None,
        next_initial_hand_target: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        # Waist ownership remains with Regular Mode for the whole hybrid skill.
        kwargs["dispatch_waist"] = False
        super().__init__(*args, **kwargs)
        self._hybrid_config_path = Path(hybrid_config_path).expanduser().resolve()
        self._hybrid_cfg, references = load_reference_images(
            self._hybrid_config_path,
            endpoint_override=hybrid_vlm_endpoint,
            model_override=hybrid_vlm_model,
        )
        self._hybrid_state = PhaseStateMachine()
        self._phase2 = Phase2Motion(self._hybrid_cfg.phase2)
        if phase3_executor not in ("vla", "rule_based"):
            raise ValueError(
                "phase3_executor must be 'vla' or 'rule_based', "
                f"got {phase3_executor!r}"
            )
        if (
            phase3_executor == "rule_based"
            and (
                self._hybrid_cfg.phase3_rule_based is None
                or next_initial_arm_target is None
                or next_initial_hand_target is None
            )
        ):
            raise ValueError(
                "phase3_executor='rule_based' requires phase3_rule_based config "
                "and the canonical next-skill arm/hand initial targets"
            )
        self._phase3_executor = phase3_executor
        self._phase3_rule = (
            None
            if phase3_executor != "rule_based"
            else Phase3RuleController(
                # The validation above guarantees all three values exist.
                self._hybrid_cfg.phase3_rule_based,
                self._hybrid_cfg.interlock,
                np.asarray(next_initial_arm_target, dtype=np.float64),
                np.asarray(next_initial_hand_target, dtype=np.float64),
            )
        )
        self._phase3_complete = False
        self._phase3_failure_reason: str | None = None
        self._boundary = GraspBoundaryDetector(
            cfg=self._hybrid_cfg.boundary,
            interlock_cfg=self._hybrid_cfg.interlock,
            vlm_cfg=self._hybrid_cfg.vlm,
            reference_images_b64=references,
        )
        self._history = FrameHistory(self._hybrid_cfg.vlm.history_frames + 1)
        left_tcp = LEFT_WRIST_TOOL_OFFSET_M - np.asarray(DEX1_BASE_OFFSET)
        right_tcp = RIGHT_WRIST_TOOL_OFFSET_M - np.asarray(DEX1_BASE_OFFSET)
        self._ik = {
            Side.LEFT: G1ArmKinematics(tcp_offset=left_tcp),
            Side.RIGHT: G1ArmKinematics(tcp_offset=right_tcp),
        }
        self._last_safe_arm: np.ndarray | None = None

    @property
    def hybrid_phase(self) -> Phase:
        return self._hybrid_state.phase

    @property
    def is_complete(self) -> bool:
        return self._phase3_executor == "rule_based" and self._phase3_complete

    @property
    def failure_reason(self) -> str | None:
        return self._phase3_failure_reason

    def _on_start(self, params: dict) -> None:
        super()._on_start(params)
        self._hybrid_state.reset()
        self._phase2.reset()
        if self._phase3_rule is not None:
            self._phase3_rule.reset()
        self._phase3_complete = False
        self._phase3_failure_reason = None
        self._boundary.reset()
        self._history.clear()
        self._last_safe_arm = None

    def _on_stop(self) -> None:
        self._history.clear()
        super()._on_stop()

    @staticmethod
    def _time_seconds(obs: dict) -> float:
        value = float(obs.get("t", time.monotonic_ns()))
        return value * 1e-9 if value > 1e12 else value

    @staticmethod
    def _joint_positions(obs: dict) -> np.ndarray:
        state = obs.get("joint_state")
        values = np.asarray(getattr(state, "position", ()), dtype=np.float64)
        if values.shape != (29,) or not np.all(np.isfinite(values)):
            raise RuntimeError("pick-leg hybrid requires a fresh finite 29-D joint state")
        return values

    @staticmethod
    def _joint_velocities(obs: dict) -> np.ndarray:
        state = obs.get("joint_state")
        values = np.asarray(getattr(state, "velocity", ()), dtype=np.float64)
        if values.shape != (29,) or not np.all(np.isfinite(values)):
            raise RuntimeError(
                "pick-leg hybrid requires a fresh finite 29-D joint velocity"
            )
        return values

    def _measured_ee(self, obs: dict) -> tuple[EePose, EePose]:
        if self._fk is None:
            raise RuntimeError("pick-leg hybrid requires G1WristFK")
        left_p, left_r, right_p, right_r = self._fk.compute_ee_transforms(
            self._joint_positions(obs)
        )
        return (
            EePose.of(left_p, _matrix_to_quat_wxyz(left_r)),
            EePose.of(right_p, _matrix_to_quat_wxyz(right_r)),
        )

    def _push_vlm_frame(self, obs: dict) -> None:
        head = np.asarray(obs["head_rgb"])
        frame = split_head_stereo(head)[0] if head.shape[:2] == (480, 1280) else head
        self._history.push(frame, obs.get("cleaned") or ())

    def step(self, obs: dict) -> np.ndarray | None:
        phase = self._hybrid_state.phase
        if phase is Phase.CARRY_TO_LEFT:
            return self._step_motion_planning(obs)
        if phase is Phase.HANDOVER_ONWARD and self._phase3_executor == "rule_based":
            return self._step_rule_phase3(obs)

        arm = super().step(obs)
        if arm is None:
            return None
        self._last_safe_arm = np.asarray(arm, dtype=np.float64).copy()
        self._push_vlm_frame(obs)

        if phase is Phase.APPROACH_GRASP:
            hand_state_obj = obs.get("hand_state")
            hand_state = np.asarray(
                getattr(hand_state_obj, "position_rad", hand_state_obj), dtype=np.float64
            )
            if self.last_action is None:
                return arm
            hand_cmd = np.asarray(self.last_action.action_chunk[0, HAND_SLICE], dtype=np.float64)
            interlock_ready = is_grasping(
                hand_state,
                hand_cmd,
                RIGHT,
                self._hybrid_cfg.interlock,
            )
            decision = self._boundary.update(
                t=self._time_seconds(obs),
                hand_state=hand_state,
                hand_cmd=hand_cmd,
                # JPEG/overlay is intentionally lazy.  Before physical grasp the
                # detector returns without consulting images, so encoding them at
                # 30 Hz would steal time from the VLA loop for no benefit.
                images_b64=(
                    self._history.as_vlm_images(
                        overlay_cfg=self._hybrid_cfg.overlay,
                        jpeg_quality=self._hybrid_cfg.vlm.jpeg_quality,
                    )
                    if interlock_ready
                    else ()
                ),
                current_phase_text="右手でテーブル脚へ接近し把持する",
            )
            metadata = dict(self.last_action.metadata)
            metadata.update({
                "hybrid_phase": phase.value,
                "hybrid_boundary_interlock": decision.interlock,
                "hybrid_vlm_called": decision.vlm_called,
                "hybrid_vlm_value": decision.vlm_value,
                "hybrid_vlm_hits": decision.hits,
                "hybrid_vlm_latency_sec": decision.latency_sec,
                "hybrid_vlm_error": decision.error,
            })
            self.last_action = PolicyAction(
                action_chunk=self.last_action.action_chunk,
                latency_ms=self.last_action.latency_ms,
                metadata=metadata,
            )
            if decision.vlm_called:
                print(
                    "[hybrid] grasp boundary: "
                    f"interlock={decision.interlock} vlm={decision.vlm_value} "
                    f"hits={decision.hits}/{self._hybrid_cfg.boundary.confirm_count} "
                    f"latency={decision.latency_sec:.3f}s"
                    + (f" error={decision.error}" if decision.error else ""),
                    file=sys.stderr,
                )
            if decision.fire:
                left, right = self._measured_ee(obs)
                self._phase2.start(self._time_seconds(obs), left, right)
                self._hybrid_state.advance()
                self._action_queue.clear()
                self._action_queue_next_index = 0
                print(
                    "[hybrid] transition: phase 1 VLA -> phase 2 Cartesian/IK",
                    file=sys.stderr,
                )
        else:
            metadata = dict(self.last_action.metadata if self.last_action else {})
            metadata["hybrid_phase"] = phase.value
            if self.last_action is not None:
                self.last_action = PolicyAction(
                    action_chunk=self.last_action.action_chunk,
                    latency_ms=self.last_action.latency_ms,
                    metadata=metadata,
                )
        return arm

    def _step_motion_planning(self, obs: dict) -> np.ndarray:
        now = self._time_seconds(obs)
        measured = self._joint_positions(obs)
        waist = measured[12:15]
        left_goal, right_goal = self._phase2.pose_at(now)
        goals = {Side.LEFT: left_goal, Side.RIGHT: right_goal}
        slices = {Side.LEFT: slice(15, 22), Side.RIGHT: slice(22, 29)}
        solved: list[np.ndarray] = []
        failures: list[str] = []
        for side in (Side.LEFT, Side.RIGHT):
            goal = goals[side]
            target = EEPose(
                position=goal.pos,
                rpy=matrix_to_rpy(_quat_to_matrix_wxyz(goal.quat)),
            )
            seed = (
                measured[slices[side]]
                if self._last_safe_arm is None
                else self._last_safe_arm[(slice(0, 7) if side is Side.LEFT else slice(7, 14))]
            )
            result = self._ik[side].ik(target, seed, side, waist)
            if not result.ok:
                failures.append(
                    f"{side.value}:{result.status.value}:"
                    f"pos={result.position_error:.4f}:rot={result.rotation_error:.4f}"
                )
            solved.append(result.q)

        requested_arm = np.concatenate(solved)
        if failures:
            # A bad task-space point must never become a partial one-arm command.
            requested_arm = (
                measured[15:29].copy()
                if self._last_safe_arm is None else self._last_safe_arm.copy()
            )

        target19 = np.zeros(ACTION_DIM_TOTAL, dtype=np.float64)
        target19[WAIST_SLICE] = waist
        target19[ARMS_SLICE] = requested_arm
        target19[HAND_SLICE] = [
            self._hybrid_cfg.phase2.hold_left,
            self._hybrid_cfg.phase2.hold_right,
        ]
        if self._teacher_range is not None:
            target19 = self._teacher_range.apply(target19).target_19d
        if self._motion_limiter is not None:
            target19 = self._motion_limiter.apply(
                target=target19, measured=self._measured_19d(obs)
            )
        self._hand_actuator.send_action(target19[HAND_SLICE].tolist())
        arm = target19[ARMS_SLICE].astype(np.float64, copy=True)
        self._last_safe_arm = arm.copy()
        self._prev_action_19d = target19.astype(np.float32, copy=True)
        observation = self._build_observation(obs)
        self._frames_prev = dict(observation.frames_bgr)
        self.last_action = PolicyAction(
            action_chunk=target19.reshape(1, ACTION_DIM_TOTAL).copy(),
            latency_ms=0.0,
            metadata={
                "hybrid_phase": Phase.CARRY_TO_LEFT.value,
                "hybrid_executor": "g1_joint_ik",
                "hybrid_phase2_progress": self._phase2.progress(now),
                "hybrid_ik_failures": failures,
                "motion_limiter_applied": self._motion_limiter is not None,
            },
        )

        actual_left, actual_right = self._measured_ee(obs)
        if not failures and self._phase2.reached(actual_left, actual_right):
            self._hybrid_state.advance()
            # Discard the pre-MP chunk/temporal state, but retain the same loaded worker.
            self._flush_execution_state()
            if self._phase3_executor == "rule_based":
                if self._phase3_rule is None:
                    raise RuntimeError("rule-based phase3 controller is unavailable")
                self._phase3_rule.start(
                    now, actual_left, actual_right, measured[15:29]
                )
            print(
                "[hybrid] transition: phase 2 Cartesian/IK -> phase 3 "
                f"{self._phase3_executor}",
                file=sys.stderr,
            )
        return arm

    def _step_rule_phase3(self, obs: dict) -> np.ndarray:
        """Run the opt-in measured handover/regrasp FSM without GR00T."""
        if self._phase3_rule is None:
            raise RuntimeError("rule-based phase3 controller is unavailable")
        now = self._time_seconds(obs)
        measured = self._joint_positions(obs)
        measured19 = self._measured_19d(obs)
        actual_left, actual_right = self._measured_ee(obs)
        hand_state = measured19[HAND_SLICE]
        previous_hand = (
            hand_state
            if self._prev_action_19d is None
            else np.asarray(self._prev_action_19d[HAND_SLICE], dtype=np.float64)
        )
        previous_stage = self._phase3_rule.stage
        previous_arm = (
            measured[15:29]
            if self._prev_action_19d is None
            else np.asarray(self._prev_action_19d[ARMS_SLICE], dtype=np.float64)
        )
        measured_arm_velocity = self._joint_velocities(obs)[15:29]
        command = self._phase3_rule.step(
            t=now,
            left=actual_left,
            right=actual_right,
            hand_state=hand_state,
            previous_hand_command=previous_hand,
            arm_state=measured[15:29],
            arm_velocity=measured_arm_velocity,
            previous_arm_command=previous_arm,
        )
        if command.stage is not previous_stage:
            print(
                "[hybrid] phase 3 rule transition: "
                f"{previous_stage.value} -> {command.stage.value}",
                file=sys.stderr,
            )

        requested_arm = (
            measured[15:29].copy()
            if self._last_safe_arm is None
            else self._last_safe_arm.copy()
        )
        failures: list[str] = []
        if command.failure_reason is None and command.arm_target is not None:
            requested_arm = command.arm_target.copy()
        elif (
            command.failure_reason is None
            and command.moving_side is not None
            and not command.complete
        ):
            side = Side.LEFT if command.moving_side == 0 else Side.RIGHT
            goal = command.left if side is Side.LEFT else command.right
            target = EEPose(
                position=goal.pos,
                rpy=matrix_to_rpy(_quat_to_matrix_wxyz(goal.quat)),
            )
            arm_slice = slice(0, 7) if side is Side.LEFT else slice(7, 14)
            result = self._ik[side].ik(
                target,
                requested_arm[arm_slice],
                side,
                measured[12:15],
            )
            if result.ok:
                requested_arm[arm_slice] = result.q
            else:
                failures.append(
                    f"{side.value}:{result.status.value}:"
                    f"pos={result.position_error:.4f}:rot={result.rotation_error:.4f}"
                )

        if failures and self._phase3_failure_reason is None:
            self._phase3_failure_reason = (
                f"phase3 rule IK failure at {command.stage.value}: "
                + ",".join(failures)
            )
        if command.failure_reason is not None:
            self._phase3_failure_reason = command.failure_reason

        target19 = measured19.copy()
        target19[ARMS_SLICE] = requested_arm
        # A failure tick must not advance either actuator one more step before
        # Orchestrator observes ``failure_reason``.  Retain the exact previous
        # transmitted Dex1 target together with the last safe arm target.
        target19[HAND_SLICE] = (
            previous_hand
            if self._phase3_failure_reason is not None
            else command.hand
        )
        if self._teacher_range is not None:
            target19 = self._teacher_range.apply(target19).target_19d
        if self._motion_limiter is not None:
            target19 = self._motion_limiter.apply(
                target=target19, measured=measured19
            )
        self._hand_actuator.send_action(target19[HAND_SLICE].tolist())
        arm = target19[ARMS_SLICE].astype(np.float64, copy=True)
        self._last_safe_arm = arm.copy()
        self._prev_action_19d = target19.astype(np.float32, copy=True)
        self.last_action = PolicyAction(
            action_chunk=target19.reshape(1, ACTION_DIM_TOTAL).copy(),
            latency_ms=0.0,
            metadata={
                "hybrid_phase": Phase.HANDOVER_ONWARD.value,
                "hybrid_executor": "rule_based_phase3",
                "hybrid_phase3_stage": command.stage.value,
                "hybrid_phase3_progress": command.progress,
                "hybrid_phase3_complete": command.complete,
                "hybrid_ik_failures": failures,
                "motion_limiter_applied": self._motion_limiter is not None,
                "hybrid_phase3_requested_arm_target": (
                    None
                    if command.arm_target is None
                    else command.arm_target.astype(float).tolist()
                ),
                "hybrid_phase3_measured_arm": measured[15:29].astype(float).tolist(),
                "hybrid_phase3_measured_arm_velocity": (
                    measured_arm_velocity.astype(float).tolist()
                ),
                "hybrid_phase3_transmitted_arm": arm.astype(float).tolist(),
                "hybrid_phase3_arm_goal_max_error_rad": (
                    None
                    if command.arm_target is None
                    else float(
                        np.max(
                            np.abs(
                                measured[15:29]
                                - command.arm_target
                            )
                        )
                    )
                ),
            },
        )
        if command.complete and not self._phase3_complete:
            self._phase3_complete = True
            print(
                "[hybrid] phase 3 rule-based handover completed at the "
                "insert_table_leg dataset initial arm/hand pose",
                file=sys.stderr,
            )
        return arm
