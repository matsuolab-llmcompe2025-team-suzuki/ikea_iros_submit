"""G1 URDF-based forward kinematics for left/right wrist EE pose (Issue #125 Phase A-5)。

# 目的

Orin `real_hw_bridge_node` は `/joint_states` (29 dim) のみ publish、`/g1/ee_state`
は real deploy launch に含まれない。両 VLA policy (RAMEN-Ori 71D / GR00T 49D REAL_G1_
RELATIVE_EEF) は `ee_state` を state 入力の一部として使う (訓練で見た spatial signal)
ため、Desktop 側で joint_positions → left/right wrist の (xyz + euler xyz) を root
frame で計算して埋める必要がある。本 module がその FK を提供する。

# 設計

- **依存**: numpy のみ (pinocchio は cmeel-boost が numpy>=2 を要求、runtime env の
  numpy==1.26.4 と非互換なので採用不可、Issue #125 で判断)。scipy も足さない
  (euler extraction は inline)。
- **Chain**: G1 URDF から pelvis → left/right_wrist_yaw_link の 10 joint chain
  (waist 3 + arm 7) を init 時に一度 parse し、各 joint の fixed origin transform と
  axis + G1_JOINT_NAMES index を cache。tick loop での重複 parse を避ける。
- **Euler convention**: training-side `g1_full_body_mapping._euler_xyz_matrix` と
  厳密一致 (Rz(yaw) @ Ry(pitch) @ Rx(roll)、scipy `Rotation.from_euler("xyz")` と同順)。
  extraction は逆算式 (gimbal lock: pitch=±π/2 で近傍縮退、実 wrist pose が
  垂直近傍を通ることは稀のため 実運用可)。
- **Reference frame**: chain root = `pelvis` (G1 URDF)。training dataset の
  `ee_state` reference_frame と一致 (BitRobot G1_WBT source_dataset.eef_pose_format
  = "root_link" = pelvis 前提、cross-verify で確認予定)。

# 使用例

```python
fk = G1WristFK.from_urdf(URDF_PATH)
joint_positions = np.array(...)  # (29,) G1_JOINT_NAMES 順
ee_state = fk.compute_ee_state(joint_positions)  # (12,) left(xyz+euler)+right(xyz+euler)
```
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


# G1 URDF (repo 内、Orin bringup と同じ file を Desktop からも参照)
DEFAULT_URDF_PATH: str = (
    "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/"
    "g1_29dof_mode_15_with_dex1_1.urdf"
)

# 参照: g1_hw_bridge/joint_mapping.py:G1_JOINT_NAMES と一致 (SDK motor index 順)
G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",       # 0
    "left_hip_roll_joint",        # 1
    "left_hip_yaw_joint",         # 2
    "left_knee_joint",            # 3
    "left_ankle_pitch_joint",     # 4
    "left_ankle_roll_joint",      # 5
    "right_hip_pitch_joint",      # 6
    "right_hip_roll_joint",       # 7
    "right_hip_yaw_joint",        # 8
    "right_knee_joint",           # 9
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
    "waist_yaw_joint",            # 12
    "waist_roll_joint",           # 13
    "waist_pitch_joint",          # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27
    "right_wrist_yaw_joint",      # 28
)

LEFT_WRIST_LINK: str = "left_wrist_yaw_link"
RIGHT_WRIST_LINK: str = "right_wrist_yaw_link"
ROOT_LINK: str = "pelvis"
# BitRobot/GR00T labels use a tool point 5 cm along wrist-yaw link +X rather
# than the URDF link origin.  Keeping this explicit avoids a systematic 5 cm
# state error on both arms.
WRIST_TOOL_OFFSET_M = np.array([0.05, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class ChainJoint:
    """1 chain step: fixed origin transform (parent → this joint frame) と rotation axis。

    Attributes:
        name: URDF joint name。
        joint_index: G1_JOINT_NAMES 内 index (0..28)、joint_positions[index] を回転角に使う。
        fixed_T: (4, 4) float64、parent link frame での joint origin transform (URDF
                 <origin xyz rpy>)。この後に joint 回転が乗る。
        axis: (3,) float64、joint rotation axis (URDF <axis xyz>、unit vector 想定)。
    """

    name: str
    joint_index: int
    fixed_T: np.ndarray  # (4, 4)
    axis: np.ndarray     # (3,)


def _euler_xyz_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler XYZ (radians) → 3x3 rotation matrix (Rz(yaw) @ Ry(pitch) @ Rx(roll))。

    scipy `Rotation.from_euler("xyz", [roll, pitch, yaw])` と一致、
    training `g1_full_body_mapping._euler_xyz_matrix` と inline 一致。
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _matrix_to_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """3x3 rotation matrix → (roll, pitch, yaw)、_euler_xyz_matrix の inverse。

    Rz(yaw)@Ry(pitch)@Rx(roll) の分解、scipy `Rotation.from_matrix(R).as_euler("xyz")`
    と一致 (gimbal lock at |pitch| ≈ π/2 で yaw=0 に固定、roll に集約する慣例)。

    Args:
        R: (3, 3) rotation matrix (proper orthogonal)。

    Returns:
        (roll, pitch, yaw) in radians。
    """
    # pitch = arcsin(-R[2,0]) だが安全のため clip
    sp = -float(R[2, 0])
    sp_clip = max(-1.0, min(1.0, sp))
    pitch = np.arcsin(sp_clip)
    if abs(sp_clip) < 1.0 - 1e-9:
        roll = np.arctan2(float(R[2, 1]), float(R[2, 2]))
        yaw = np.arctan2(float(R[1, 0]), float(R[0, 0]))
    else:
        # gimbal lock: yaw を 0 に固定、roll に集約 (scipy 慣例)
        yaw = 0.0
        roll = np.arctan2(-float(R[1, 2]), float(R[1, 1]))
    return float(roll), float(pitch), float(yaw)


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula: axis (3,) + angle (rad) → 3x3 rotation matrix。"""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    a = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    x, y, z = float(a[0]), float(a[1]), float(a[2])
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _make_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _parse_vec3(s: str, default: str = "0 0 0") -> np.ndarray:
    """URDF attribute の "x y z" 文字列 → (3,) float64。"""
    parts = (s or default).split()
    if len(parts) != 3:
        raise ValueError(f"expected 3 values, got {s!r}")
    return np.array([float(p) for p in parts], dtype=np.float64)


def _parse_urdf_chain(
    urdf_path: str | Path, target_link: str, joint_name_to_index: dict[str, int]
) -> list[ChainJoint]:
    """URDF root (pelvis) → target_link の revolute chain を返す。

    Raises:
        ValueError: chain 内に revolute 以外の joint がある / joint が G1_JOINT_NAMES に
                    含まれない場合。
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # child_link → (joint_name, jtype, parent, xyz, rpy, axis)
    child_map: dict[str, tuple[str, str, str, str, str, str | None]] = {}
    for j in root.findall("joint"):
        name = j.get("name") or ""
        jtype = j.get("type") or ""
        parent_el = j.find("parent")
        child_el = j.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link") or ""
        child = child_el.get("link") or ""
        origin = j.find("origin")
        xyz = (origin.get("xyz") if origin is not None else None) or "0 0 0"
        rpy = (origin.get("rpy") if origin is not None else None) or "0 0 0"
        axis_el = j.find("axis")
        axis = axis_el.get("xyz") if axis_el is not None else None
        child_map[child] = (name, jtype, parent, xyz, rpy, axis)

    # target から root へ遡って chain 作る
    chain_reversed: list[ChainJoint] = []
    cur = target_link
    while cur in child_map:
        name, jtype, parent, xyz_s, rpy_s, axis_s = child_map[cur]
        if jtype != "revolute":
            raise ValueError(
                f"chain joint {name!r} has type {jtype!r}, expected 'revolute'"
            )
        if name not in joint_name_to_index:
            raise ValueError(
                f"chain joint {name!r} not in G1_JOINT_NAMES (SDK joint layout)"
            )
        if axis_s is None:
            raise ValueError(f"chain joint {name!r} has no <axis xyz>")
        xyz = _parse_vec3(xyz_s)
        rpy = _parse_vec3(rpy_s)
        # fixed transform: translate(xyz) @ Rz(yaw) @ Ry(pitch) @ Rx(roll)
        R_fixed = _euler_xyz_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        fixed_T = _make_4x4(R_fixed, xyz)
        axis = _parse_vec3(axis_s)
        chain_reversed.append(
            ChainJoint(
                name=name,
                joint_index=joint_name_to_index[name],
                fixed_T=fixed_T,
                axis=axis,
            )
        )
        cur = parent
    if cur != ROOT_LINK:
        raise ValueError(
            f"chain from {target_link!r} did not reach {ROOT_LINK!r} (stopped at {cur!r})"
        )
    return list(reversed(chain_reversed))


class G1WristFK:
    """G1 URDF 由来 pelvis frame 基準の left/right wrist FK (pure numpy)。

    init 時に URDF から chain を一度 parse し、per-tick は 10 joint × 2 chain の
    4x4 matrix chain 積で ee_state を計算 (~20 4x4 matmul、numpy で 100μs order)。

    Attributes:
        _left_chain: (list[ChainJoint]) pelvis → left_wrist_yaw_link chain。
        _right_chain: pelvis → right_wrist_yaw_link chain。
    """

    def __init__(self, left_chain: Sequence[ChainJoint], right_chain: Sequence[ChainJoint]):
        self._left_chain = tuple(left_chain)
        self._right_chain = tuple(right_chain)

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path = DEFAULT_URDF_PATH,
        joint_names: Sequence[str] = G1_JOINT_NAMES,
    ) -> "G1WristFK":
        """URDF file を parse して G1WristFK instance を返す。

        Args:
            urdf_path: URDF file path (repo 内 default = g1_description/urdf/...).
            joint_names: SDK joint layout (index 0..28 に対応する URDF joint name 順)。

        Returns:
            G1WristFK。
        """
        joint_name_to_index = {name: i for i, name in enumerate(joint_names)}
        left_chain = _parse_urdf_chain(urdf_path, LEFT_WRIST_LINK, joint_name_to_index)
        right_chain = _parse_urdf_chain(urdf_path, RIGHT_WRIST_LINK, joint_name_to_index)
        return cls(left_chain=left_chain, right_chain=right_chain)

    def _fk_chain(
        self, chain: Sequence[ChainJoint], joint_positions: np.ndarray
    ) -> np.ndarray:
        """chain 経由で pelvis → target_link の 4x4 transform を計算。"""
        T = np.eye(4, dtype=np.float64)
        for cj in chain:
            angle = float(joint_positions[cj.joint_index])
            R_joint = _axis_angle_matrix(cj.axis, angle)
            T_joint = _make_4x4(R_joint, np.zeros(3, dtype=np.float64))
            # chain step: T = T @ fixed_origin @ joint_rotation
            T = T @ cj.fixed_T @ T_joint
        return T

    def compute_ee_state(self, joint_positions: np.ndarray) -> np.ndarray:
        """(29,) joint_positions → (12,) ee_state = left(xyz+euler) + right(xyz+euler)。

        Args:
            joint_positions: (29,) float、G1_JOINT_NAMES 順 (SDK motor index 順)。
                             leg 部分 [0:12] は wrist FK に不要だが shape 検証で要求。

        Returns:
            (12,) float32、[left_x, left_y, left_z, left_roll, left_pitch, left_yaw,
                            right_x, right_y, right_z, right_roll, right_pitch, right_yaw]。
        """
        jp = np.asarray(joint_positions, dtype=np.float64)
        if jp.shape != (29,):
            raise ValueError(f"joint_positions must be (29,), got {jp.shape}")
        T_left = self._fk_chain(self._left_chain, jp)
        T_right = self._fk_chain(self._right_chain, jp)

        left_xyz = T_left[:3, 3] + T_left[:3, :3] @ WRIST_TOOL_OFFSET_M
        left_euler = _matrix_to_euler_xyz(T_left[:3, :3])
        right_xyz = T_right[:3, 3] + T_right[:3, :3] @ WRIST_TOOL_OFFSET_M
        right_euler = _matrix_to_euler_xyz(T_right[:3, :3])

        out = np.concatenate(
            [
                left_xyz,
                np.asarray(left_euler, dtype=np.float64),
                right_xyz,
                np.asarray(right_euler, dtype=np.float64),
            ]
        ).astype(np.float32, copy=False)
        return out

    def compute_ee_transforms(
        self, joint_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(29,) joint_positions → 左右 tool-point の (pos(3), R(3x3))。

        `compute_ee_state` と同じ pelvis frame・同じ 5cm tool offset を使うが、
        回転を euler ではなく **回転行列そのまま**で返す。task-space adapter が
        matrix→quat 直変換 (gimbal lock 縮退回避) するための下位 API。

        Args:
            joint_positions: (29,) float、G1_JOINT_NAMES 順。

        Returns:
            (left_pos(3), left_R(3x3), right_pos(3), right_R(3x3)) 全て float64。
            pos は wrist tool point (pelvis frame)、R は wrist-yaw link の姿勢。
        """
        jp = np.asarray(joint_positions, dtype=np.float64)
        if jp.shape != (29,):
            raise ValueError(f"joint_positions must be (29,), got {jp.shape}")
        T_left = self._fk_chain(self._left_chain, jp)
        T_right = self._fk_chain(self._right_chain, jp)
        left_R = T_left[:3, :3].copy()
        right_R = T_right[:3, :3].copy()
        left_pos = T_left[:3, 3] + left_R @ WRIST_TOOL_OFFSET_M
        right_pos = T_right[:3, 3] + right_R @ WRIST_TOOL_OFFSET_M
        return left_pos, left_R, right_pos, right_R
