"""Pick-leg GR00T N1.7 relative-EEF adapter (four cameras, arm IK, Dex1).

The official GR00T postprocessor converts the 20-D relative action to an
absolute pelvis-frame EEF target. This adapter then solves the two 7-DoF arms
without forwarding the model's whole-body state to the robot.
"""

from __future__ import annotations

import numpy as np

from inference.desktop.lower_policy.kinematics.g1_arm import (
    DEX1_BASE_OFFSET,
    G1ArmKinematics,
)
from inference.desktop.lower_policy.kinematics.types import (
    EEPose,
    Side,
    matrix_to_rpy,
    rpy_to_matrix,
)
from inference.desktop.lower_policy.policies.base import (
    Observation,
    PolicyAction,
    PolicyConfig,
    RawRobotState,
)
from inference.desktop.lower_policy.policies.groot import (
    _wrist_pose_to_xyz_rot6d,
)
from inference.desktop.lower_policy.policies.groot_pick_legs import (
    CAMERAS,
    DEFAULT_LANGUAGE_PROMPT,
    DEX1_DATASET_MAX_RAD,
    _PickLegsWorkerClient,
    build_state_from_raw as build_joint_pick_state,
    validate_pick_legs_config,
)
from inference.desktop.perception.g1_urdf_fk import (
    LEFT_WRIST_TOOL_OFFSET_M,
    RIGHT_WRIST_TOOL_OFFSET_M,
)

STATE_DIM = 56
RAW_ACTION_DIM = 20
ACTION_DIM = 19
EXECUTION_HORIZON = 4


def build_state_from_raw(raw: RawRobotState) -> np.ndarray:
    if raw.ee_state.shape != (12,) or not np.isfinite(raw.ee_state).all():
        raise ValueError("relative-EEF pick requires finite left/right 12-D EE state")
    body_and_hand = build_joint_pick_state(raw)
    result = np.concatenate(
        (
            _wrist_pose_to_xyz_rot6d(raw.ee_state[:6]),
            _wrist_pose_to_xyz_rot6d(raw.ee_state[6:]),
            body_and_hand[36:38],
            body_and_hand[:36],
        )
    ).astype(np.float32)
    if result.shape != (STATE_DIM,) or not np.isfinite(result).all():
        raise ValueError("relative-EEF pick state must be finite 56-D")
    return result


def rot6d_rows_to_matrix(values: np.ndarray) -> np.ndarray:
    rows = np.array(values, dtype=np.float64, copy=True)
    if rows.shape != (6,) or not np.isfinite(rows).all():
        raise ValueError("EEF rotation must be finite 6-D")
    first = rows[:3]
    n0 = float(np.linalg.norm(first))
    if n0 < 1e-6:
        raise ValueError("EEF rotation has a degenerate first row")
    first /= n0
    second = rows[3:] - first * float(np.dot(first, rows[3:]))
    n1 = float(np.linalg.norm(second))
    if n1 < 1e-6:
        raise ValueError("EEF rotation has a degenerate second row")
    second /= n1
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=0)


def _pose_from_xyz_rot6d(values: np.ndarray) -> EEPose:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (9,) or not np.isfinite(vector).all():
        raise ValueError("EEF pose must be finite 9-D")
    return EEPose(
        position=vector[:3].copy(),
        rpy=matrix_to_rpy(rot6d_rows_to_matrix(vector[3:])),
    )


class _EERelativePickWorkerClient(_PickLegsWorkerClient):
    STATE_DIM = STATE_DIM
    RAW_ACTION_DIM = RAW_ACTION_DIM
    MODEL_KIND = "ee_relative"

    def __init__(self, cfg: PolicyConfig) -> None:
        left_tcp = LEFT_WRIST_TOOL_OFFSET_M - np.asarray(DEX1_BASE_OFFSET)
        right_tcp = RIGHT_WRIST_TOOL_OFFSET_M - np.asarray(DEX1_BASE_OFFSET)
        self._ik = {
            Side.LEFT: G1ArmKinematics(tcp_offset=left_tcp, limit_rad=2.1),
            Side.RIGHT: G1ArmKinematics(tcp_offset=right_tcp, limit_rad=2.1),
        }
        super().__init__(cfg)

    def warmup_without_ik(self) -> None:
        """Prime the GPU before arm ownership, without inventing joint targets."""
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("relative-EEF worker pipes are closed")
        self._request_id += 1
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[3:9] = (1, 0, 0, 0, 1, 0)
        state[12:18] = (1, 0, 0, 0, 1, 0)
        state[18:20] = DEX1_DATASET_MAX_RAD
        state[22:24] = (0.70, 1.0)
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        cameras = {
            f"observation.images.cam_{i}": self._jpeg(black, f"cam_{i}")
            for i in range(4)
        }
        self._send_message(self._process.stdin, {
            "type": "predict", "request_id": self._request_id,
            "state": state, "cameras": cameras,
            "task": DEFAULT_LANGUAGE_PROMPT,
        })
        response = self._receive_message(self._process.stdout)
        if (
            not isinstance(response, dict)
            or response.get("type") != "prediction"
            or response.get("request_id") != self._request_id
        ):
            raise RuntimeError(f"relative-EEF GPU warmup failed: {response!r}")
        action = np.asarray(response.get("actions"), dtype=np.float32)
        if action.shape != (16, RAW_ACTION_DIM) or not np.isfinite(action).all():
            raise RuntimeError("relative-EEF GPU warmup returned invalid action")

    def _decode_actions(
        self, raw: np.ndarray, obs: Observation, latency_ms: float,
    ) -> PolicyAction:
        if obs.state.shape != (STATE_DIM,):
            raise ValueError("relative-EEF pick observation must be 56-D")
        # state=[left_ee9,right_ee9,hand2,root7,body29].
        body = obs.state[27:56]
        waist = np.asarray(body[12:15], dtype=np.float64)
        seeds = {
            Side.LEFT: np.asarray(body[15:22], dtype=np.float64).copy(),
            Side.RIGHT: np.asarray(body[22:29], dtype=np.float64).copy(),
        }
        # Verify that the live state and the IK implementation share the same
        # pelvis frame and tool point before allowing any target to pass.
        for side, state_slice in ((Side.LEFT, slice(0, 9)), (Side.RIGHT, slice(9, 18))):
            measured = _pose_from_xyz_rot6d(obs.state[state_slice])
            fk = self._ik[side].fk(seeds[side], side, waist)
            pos_error = float(np.linalg.norm(measured.position - fk.position))
            rot_error = float(np.linalg.norm(
                rpy_to_matrix(measured.rpy) - rpy_to_matrix(fk.rpy)
            ))
            if pos_error > 0.03 or rot_error > 0.18:
                raise RuntimeError(
                    f"relative-EEF pick FK/IK frame mismatch on {side.value}: "
                    f"position={pos_error:.3f}m rotation_matrix={rot_error:.3f}"
                )
        commands: list[np.ndarray] = []
        for row in raw[:EXECUTION_HORIZON]:
            solved: dict[Side, np.ndarray] = {}
            for side, action_slice in (
                (Side.LEFT, slice(0, 9)), (Side.RIGHT, slice(9, 18)),
            ):
                target = _pose_from_xyz_rot6d(row[action_slice])
                result = self._ik[side].ik(target, seeds[side], side, waist)
                if not result.ok:
                    raise RuntimeError(
                        f"relative-EEF pick IK failed on {side.value}: "
                        f"{result.status.value}, position_error="
                        f"{result.position_error:.3f}m"
                    )
                solved[side] = result.q.copy()
            seeds.update(solved)
            command = np.concatenate((
                waist, solved[Side.LEFT], solved[Side.RIGHT],
                np.clip(row[18:20], 0.0, DEX1_DATASET_MAX_RAD),
            ))
            if command.shape != (ACTION_DIM,) or not np.isfinite(command).all():
                raise RuntimeError("relative-EEF pick IK produced invalid 19-D target")
            commands.append(command)
        return PolicyAction(
            action_chunk=np.asarray(commands, dtype=np.float32),
            latency_ms=latency_ms,
            metadata={
                "mode": "none",
                "chunk_len": len(commands),
                "action_dim": ACTION_DIM,
                "decoded_action_shape_20d": tuple(raw.shape),
                "lower_body_command_dimensions": 0,
                "ik_frame": "pelvis",
            },
        )


class Gr00tPolicyPickLegsEERel:
    STATE_DIM = STATE_DIM
    ACTION_DIM = ACTION_DIM
    CHUNK_LEN = 16
    CAMERAS = CAMERAS
    DEFAULT_LANGUAGE_PROMPT = DEFAULT_LANGUAGE_PROMPT
    EXECUTION_HORIZON = EXECUTION_HORIZON
    build_state_from_raw = staticmethod(build_state_from_raw)

    def __init__(
        self, cfg: PolicyConfig, worker: _EERelativePickWorkerClient,
    ) -> None:
        self.cfg = cfg
        self._worker = worker

    @classmethod
    def from_ckpt(cls, cfg: PolicyConfig) -> "Gr00tPolicyPickLegsEERel":
        validate_pick_legs_config(cfg)
        return cls(cfg, _EERelativePickWorkerClient(cfg))

    def warmup(self, n_iter: int = 1) -> None:
        # A synthetic state cannot define meaningful arm IK, but it can safely
        # initialize GPU kernels before the final physical actuation gate.
        for _ in range(n_iter):
            self._worker.warmup_without_ik()

    def predict(self, obs: Observation) -> PolicyAction:
        return self._worker.predict(obs)

    def close(self) -> None:
        self._worker.close()
