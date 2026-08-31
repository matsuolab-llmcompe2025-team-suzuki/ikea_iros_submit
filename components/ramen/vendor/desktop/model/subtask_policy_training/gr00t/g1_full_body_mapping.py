"""Map the official G1 dataset schema to the GR00T N1.7 REAL_G1 slots."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

try:
    from .dex1_hand_synergy import dex1_to_hand
except ImportError:  # Loaded directly by data-materialization utilities.
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from gr00t.dex1_hand_synergy import dex1_to_hand


REAL_G1_RELATIVE_EEF_STATE_DIM = 49
REAL_G1_RELATIVE_EEF_ACTION_DIM = 53
REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG = "real_g1_relative_eef_relative_joints"
REAL_G1_RELATIVE_EEF_EMBODIMENT_ID = 25
GROOT_N17_NATIVE_ACTION_HORIZON = 40
SOURCE_ROBOT_Q_DIM = 36
SOURCE_ROOT_POSE_DIM = 7
SOURCE_EEF_DIM = 12
SOURCE_EEF_POSE_DIM = 6
SOURCE_STATE_DIM = SOURCE_ROBOT_Q_DIM + 2
SOURCE_ACTION_DIM = SOURCE_ROBOT_Q_DIM + 2
STANDARD_POLICY_VIDEO_KEYS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)

G1_FULL_BODY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

G1_FULL_BODY_STATE_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "left_hand": (22, 29),
    "right_arm": (29, 36),
    "right_hand": (36, 43),
}

SOURCE_JOINT_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "right_arm": (22, 29),
}

# This is the exact flattened group order in the GR00T N1.7 base checkpoint's
# real_g1_relative_eef_relative_joints processor_config.json.
REAL_G1_RELATIVE_EEF_STATE_SLICES = {
    "left_wrist_eef_9d": (0, 9),
    "right_wrist_eef_9d": (9, 18),
    "left_hand": (18, 25),
    "right_hand": (25, 32),
    "left_arm": (32, 39),
    "right_arm": (39, 46),
    "waist": (46, 49),
}

REAL_G1_RELATIVE_EEF_ACTION_SLICES = {
    "left_wrist_eef_9d": (0, 9),
    "right_wrist_eef_9d": (9, 18),
    "left_hand": (18, 25),
    "right_hand": (25, 32),
    "left_arm": (32, 39),
    "right_arm": (39, 46),
    "waist": (46, 49),
    "base_height_command": (49, 50),
    "navigate_command": (50, 53),
}

REAL_G1_RELATIVE_EEF_ACTION_CONFIGS = {
    "left_wrist_eef_9d": {
        "rep": "RELATIVE",
        "type": "EEF",
        "format": "XYZ_ROT6D",
        "state_key": "left_wrist_eef_9d",
    },
    "right_wrist_eef_9d": {
        "rep": "RELATIVE",
        "type": "EEF",
        "format": "XYZ_ROT6D",
        "state_key": "right_wrist_eef_9d",
    },
    "left_hand": {
        "rep": "ABSOLUTE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": "left_hand",
    },
    "right_hand": {
        "rep": "ABSOLUTE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": "right_hand",
    },
    "left_arm": {
        "rep": "RELATIVE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": "left_arm",
    },
    "right_arm": {
        "rep": "RELATIVE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": "right_arm",
    },
    "waist": {
        "rep": "ABSOLUTE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": "waist",
    },
    "base_height_command": {
        "rep": "ABSOLUTE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": "base_height_command",
    },
    "navigate_command": {
        "rep": "ABSOLUTE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": "navigate_command",
    },
}

UPPER_BODY_SOURCE_INDEX_MAP = (
    list(
        range(
            SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["waist"][0],
            SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["waist"][1],
        )
    )
    + list(
        range(
            SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["left_arm"][0],
            SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["left_arm"][1],
        )
    )
    + list(
        range(
            SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["right_arm"][0],
            SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["right_arm"][1],
        )
    )
    + [SOURCE_ROBOT_Q_DIM, SOURCE_ROBOT_Q_DIM + 1]
)
UPPER_BODY_STATE_DIM = len(UPPER_BODY_SOURCE_INDEX_MAP)
UPPER_BODY_ACTION_DIM = UPPER_BODY_STATE_DIM


def map_source_row_to_real_g1_relative_eef(
    *,
    ee_state: list[Any],
    ee_action: list[Any],
    robot_q_current: list[Any],
    robot_q_desired: list[Any],
    hand_state: list[Any],
    hand_cmd: list[Any],
) -> tuple[list[float], list[float]]:
    """Map BitRobot dual labels into the official GR00T N1.7 REAL_G1 slots.

    EEF and arm targets remain absolute in the materialized LeRobot rows. The
    GR00T processor converts every future target in an action chunk relative to
    the chunk's current observation. Pre-converting each row here would anchor
    future chunk elements to the wrong state.
    """
    _require_dim("ee_action", ee_action, SOURCE_EEF_DIM)
    _require_dim("robot_q_desired", robot_q_desired, SOURCE_ROBOT_Q_DIM)
    _require_dim("hand_cmd", hand_cmd, 2)

    state = map_source_state_to_real_g1_relative_eef(
        ee_state=ee_state,
        robot_q_current=robot_q_current,
        hand_state=hand_state,
    )
    action = [0.0] * REAL_G1_RELATIVE_EEF_ACTION_DIM

    for side, source_start in (("left", 0), ("right", SOURCE_EEF_POSE_DIM)):
        source_slice = slice(source_start, source_start + SOURCE_EEF_POSE_DIM)
        _copy_into(
            action,
            REAL_G1_RELATIVE_EEF_ACTION_SLICES[f"{side}_wrist_eef_9d"],
            source_euler_xyz_pose_to_xyz_rot6d(ee_action[source_slice]),
        )

    desired_joints = robot_q_desired[SOURCE_ROOT_POSE_DIM:]
    for group in ("left_arm", "right_arm", "waist"):
        source_slice = slice(*SOURCE_JOINT_SLICES[group])
        _copy_into(
            action,
            REAL_G1_RELATIVE_EEF_ACTION_SLICES[group],
            desired_joints[source_slice],
        )

    for index, side in enumerate(("left", "right")):
        _copy_into(
            action,
            REAL_G1_RELATIVE_EEF_ACTION_SLICES[f"{side}_hand"],
            dex1_to_hand(hand_cmd[index], side=side, kind="action"),
        )

    return state, action


def map_source_state_to_real_g1_relative_eef(
    *,
    ee_state: list[Any],
    robot_q_current: list[Any],
    hand_state: list[Any],
) -> list[float]:
    """Map one observed G1 state into the pinned 49-D N1.7 slot order."""

    _require_dim("ee_state", ee_state, SOURCE_EEF_DIM)
    _require_dim("robot_q_current", robot_q_current, SOURCE_ROBOT_Q_DIM)
    _require_dim("hand_state", hand_state, 2)
    state = [0.0] * REAL_G1_RELATIVE_EEF_STATE_DIM
    for side, source_start in (("left", 0), ("right", SOURCE_EEF_POSE_DIM)):
        source_slice = slice(source_start, source_start + SOURCE_EEF_POSE_DIM)
        _copy_into(
            state,
            REAL_G1_RELATIVE_EEF_STATE_SLICES[f"{side}_wrist_eef_9d"],
            source_euler_xyz_pose_to_xyz_rot6d(ee_state[source_slice]),
        )
    current_joints = robot_q_current[SOURCE_ROOT_POSE_DIM:]
    for group in ("left_arm", "right_arm", "waist"):
        source_slice = slice(*SOURCE_JOINT_SLICES[group])
        _copy_into(
            state,
            REAL_G1_RELATIVE_EEF_STATE_SLICES[group],
            current_joints[source_slice],
        )
    for index, side in enumerate(("left", "right")):
        _copy_into(
            state,
            REAL_G1_RELATIVE_EEF_STATE_SLICES[f"{side}_hand"],
            dex1_to_hand(hand_state[index], side=side, kind="state"),
        )
    return state


def source_euler_xyz_pose_to_xyz_rot6d(pose: list[Any]) -> list[float]:
    """Convert source [x,y,z,roll,pitch,yaw] radians to XYZ + row ROT6D."""
    _require_dim("source EEF pose", pose, SOURCE_EEF_POSE_DIM)
    x, y, z, roll, pitch, yaw = (float(value) for value in pose)
    rotation = _euler_xyz_matrix(roll, pitch, yaw)
    return [x, y, z, *rotation[0], *rotation[1]]


def absolute_eef_xyz_rot6d_to_relative(
    current: list[Any], target: list[Any]
) -> list[float]:
    """Return inv(T_current) @ T_target encoded as XYZ + row ROT6D."""
    _require_dim("current EEF", current, 9)
    _require_dim("target EEF", target, 9)
    current_f = [float(value) for value in current]
    target_f = [float(value) for value in target]
    current_rotation = _rot6d_matrix(current_f[3:])
    target_rotation = _rot6d_matrix(target_f[3:])
    current_rotation_t = _transpose3(current_rotation)
    relative_rotation = _matmul3(current_rotation_t, target_rotation)
    delta = [target_f[index] - current_f[index] for index in range(3)]
    relative_translation = _matvec3(current_rotation_t, delta)
    return [*relative_translation, *relative_rotation[0], *relative_rotation[1]]


def build_real_g1_relative_eef_modality_json(video_keys: list[str]) -> dict[str, Any]:
    selected_video_keys = standard_policy_video_keys(video_keys)
    return {
        "state": {
            key: _vector_modality(start, end)
            for key, (start, end) in REAL_G1_RELATIVE_EEF_STATE_SLICES.items()
        },
        "action": {
            key: _vector_modality(start, end)
            for key, (start, end) in REAL_G1_RELATIVE_EEF_ACTION_SLICES.items()
        },
        "video": {
            key.removeprefix("observation.images."): {"original_key": key}
            for key in selected_video_keys
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
        "meta": {
            "embodiment_tag": REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG,
            "source_eef_format": "left_then_right_xyz_euler_xyz_radians_root_frame",
            "training_eef_format": "xyz_rot6d_absolute_rows; processor converts chunks with inv(T_current)@T_target",
            "action_configs": REAL_G1_RELATIVE_EEF_ACTION_CONFIGS,
            "selected_video_keys": selected_video_keys,
            "dex1_hand_mapping": "fixed official G1 seven-joint synergy with least-squares inverse",
        },
    }


def _euler_xyz_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Rz(yaw) @ Ry(pitch) @ Rx(roll), matching scipy Rotation.from_euler("xyz").
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rot6d_matrix(values: list[float]) -> list[list[float]]:
    _require_dim("ROT6D", values, 6)
    row0 = _normalize3(values[0:3])
    projection = sum(row0[index] * values[index + 3] for index in range(3))
    row1 = _normalize3([values[index + 3] - projection * row0[index] for index in range(3)])
    row2 = [
        row0[1] * row1[2] - row0[2] * row1[1],
        row0[2] * row1[0] - row0[0] * row1[2],
        row0[0] * row1[1] - row0[1] * row1[0],
    ]
    return [row0, row1, row2]


def _normalize3(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 1e-12:
        raise ValueError("rotation basis vector has near-zero norm")
    return [float(value) / norm for value in values]


def _transpose3(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _matmul3(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _matvec3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3)]


def _require_dim(name: str, values: list[Any], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"expected {expected}-D {name}, got {len(values)}")
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} contains NaN or Inf")


def map_robot_q_row_to_upper_body(row: dict[str, Any]) -> dict[str, Any]:
    """Drop root and lower-body targets for non-walking upper-body policy training."""
    state = row.get("observation.state")
    action = row.get("action")
    if state is None:
        raise ValueError("observation.state is required")
    if action is None:
        raise ValueError("action is required")
    _require_dim("observation.state", state, SOURCE_STATE_DIM)
    _require_dim("action", action, SOURCE_ACTION_DIM)

    mapped = dict(row)
    mapped["observation.state"] = [float(state[index]) for index in UPPER_BODY_SOURCE_INDEX_MAP]
    mapped["action"] = [float(action[index]) for index in UPPER_BODY_SOURCE_INDEX_MAP]
    return mapped


def standard_policy_video_keys(video_keys: list[str]) -> list[str]:
    available = set(video_keys)
    return [key for key in STANDARD_POLICY_VIDEO_KEYS if key in available]


def _copy_into(target: list[float], span: tuple[int, int], source: list[Any]) -> None:
    start, end = span
    if len(source) > end - start:
        raise ValueError(f"source length {len(source)} exceeds target span {span}")
    for offset, value in enumerate(source):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"source value for target span {span} contains NaN or Inf")
        target[start + offset] = converted


def _vector_modality(start: int, end: int) -> dict[str, Any]:
    return {"start": start, "end": end}


UPPER_BODY_JOINT_NAMES = (
    G1_FULL_BODY_JOINT_NAMES[slice(*G1_FULL_BODY_STATE_SLICES["waist"])]
    + G1_FULL_BODY_JOINT_NAMES[slice(*G1_FULL_BODY_STATE_SLICES["left_arm"])]
    + G1_FULL_BODY_JOINT_NAMES[slice(*G1_FULL_BODY_STATE_SLICES["right_arm"])]
)
UPPER_BODY_STATE_NAMES = UPPER_BODY_JOINT_NAMES + [
    "left_gripper_q",
    "right_gripper_q",
]
UPPER_BODY_ACTION_NAMES = [f"{name}_target" for name in UPPER_BODY_JOINT_NAMES] + [
    "left_gripper_q_cmd",
    "right_gripper_q_cmd",
]

EEF_9D_COMPONENT_NAMES = [
    "x",
    "y",
    "z",
    "rot6d_row0_x",
    "rot6d_row0_y",
    "rot6d_row0_z",
    "rot6d_row1_x",
    "rot6d_row1_y",
    "rot6d_row1_z",
]
DEX1_PADDED_HAND_COMPONENT_NAMES = [
    "dex1_open_close",
    "unavailable_1",
    "unavailable_2",
    "unavailable_3",
    "unavailable_4",
    "unavailable_5",
    "unavailable_6",
]


def _real_g1_group_names(*, action: bool) -> list[str]:
    names: list[str] = []
    slices = REAL_G1_RELATIVE_EEF_ACTION_SLICES if action else REAL_G1_RELATIVE_EEF_STATE_SLICES
    for group, (start, end) in slices.items():
        if group.endswith("eef_9d"):
            components = EEF_9D_COMPONENT_NAMES
        elif group in {"left_hand", "right_hand"}:
            components = DEX1_PADDED_HAND_COMPONENT_NAMES
        elif group == "left_arm":
            components = G1_FULL_BODY_JOINT_NAMES[slice(*G1_FULL_BODY_STATE_SLICES["left_arm"])]
        elif group == "right_arm":
            components = G1_FULL_BODY_JOINT_NAMES[slice(*G1_FULL_BODY_STATE_SLICES["right_arm"])]
        elif group == "waist":
            components = G1_FULL_BODY_JOINT_NAMES[slice(*G1_FULL_BODY_STATE_SLICES["waist"])]
        elif group == "base_height_command":
            components = ["value"]
        elif group == "navigate_command":
            components = ["vx", "vy", "yaw"]
        else:  # pragma: no cover - guarded by the static contract above.
            raise KeyError(group)
        if len(components) != end - start:
            raise ValueError(f"component names for {group} do not match slice {(start, end)}")
        suffix = "target" if action else "state"
        names.extend(f"{group}.{component}.{suffix}" for component in components)
    return names


REAL_G1_RELATIVE_EEF_STATE_NAMES = _real_g1_group_names(action=False)
REAL_G1_RELATIVE_EEF_ACTION_NAMES = _real_g1_group_names(action=True)

if len(REAL_G1_RELATIVE_EEF_STATE_NAMES) != REAL_G1_RELATIVE_EEF_STATE_DIM:
    raise ValueError("REAL_G1 relative-EEF state names must match the 49-D state layout")
if len(REAL_G1_RELATIVE_EEF_ACTION_NAMES) != REAL_G1_RELATIVE_EEF_ACTION_DIM:
    raise ValueError("REAL_G1 relative-EEF action names must match the 53-D action layout")
if len(UPPER_BODY_STATE_NAMES) != UPPER_BODY_STATE_DIM:
    raise ValueError("upper-body state names must match the upper-body state dimension")
if len(UPPER_BODY_ACTION_NAMES) != UPPER_BODY_ACTION_DIM:
    raise ValueError("upper-body action names must match the upper-body action dimension")
