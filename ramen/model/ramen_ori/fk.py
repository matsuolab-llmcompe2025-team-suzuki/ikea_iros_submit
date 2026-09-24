"""RAMEN-Ori 用 URDF-based FK (Issue #129 Phase B、2026-08-31)。

L4 (FK anchor loss) のため、predicted action `q_pred` (arms 14 + hand 2 = 16D、
+ teacher waist 3D を concat した 19D 相当) を URDF FK に通して両手 wrist の 3D 位置
(pelvis frame) を出す。gradient は autograd で伝播、joint 角度誤差 → 3D 位置誤差の
chain rule が自然に成立、"手が届く action" を loss 側から強制する。

# model/inference 分離方針 (user 決定 2026-08-31)

本 module は **`inference/desktop/perception/g1_urdf_fk*.py` を参考にした完全 self-contained
版**。`inference/` からの import は一切なし、URDF file も `model/ramen_ori/assets/urdf/`
に copy 済み。将来 inference 側 FK refactor に影響されない = model/ramen_ori/ 単独で
completeness を保つ。

# API

    fk = G1WristFKTorch.from_default_urdf()           # 初期化 (URDF parse は init 1 回のみ)
    q_full_19d: torch.Tensor  # shape (..., 19)、waist 3 + left_arm 7 + right_arm 7 + gripper 2
    left_pos, right_pos = fk(q_full_19d)              # 各 (..., 3) tensor、pelvis frame
    # gripper 部分 (index 17,18) は wrist chain 不参加、値は無視される
    parts = fk.forward_detailed(q_full_19d)           # 肘の位置・手先の位置・手の向き (rot6d)、左右
    feats = fk.features(q_full_19d)                   # (..., 27) FK の loss 用 (FK_FEATURE_SLICES)

# 19D layout convention

RAMEN-Ori の 16D action (arms 14 + hand 2、waist 除外) は L4 計算時に teacher waist 3D
と concat して 19D にし、本 FK に渡す:
    q_19 = torch.cat([teacher_waist_3, ramen_arms_hand_16], dim=-1)
    layout = waist(0:3) + left_arm(3:10) + right_arm(10:17) + gripper(17:19)

これは GR00T 19D layout と同一 (model.subtask_policy_training.joint_layout.GROOT_ACTION_NAMES)、
teacher waist は data_lerobot が action.robot_q_desired[:, waist_indices] から供給する。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants (自 copy、`inference/` からの import なし)
# ---------------------------------------------------------------------------

# URDF file (Issue #129 Phase B で inference/orin/... から copy)
# __file__ = model/ramen_ori/fk.py、parents[0] = model/ramen_ori/
DEFAULT_URDF_PATH: str = str(
    Path(__file__).resolve().parent / "assets" / "urdf" / "g1_29dof_mode_15_with_dex1_1.urdf"
)

# G1 SDK 29-joint order (`inference/desktop/perception/g1_urdf_fk.py::G1_JOINT_NAMES` と一致)
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

# Issue #134: Dex1 grasp point tool offset (wrist_yaw_link frame)。
# inference 側 `inference/desktop/perception/g1_urdf_fk.py::{LEFT,RIGHT}_WRIST_TOOL_OFFSET_M`
# と同一値を duplicate 定義 (self-contained 方針、user 決定 2026-08-31)。BitRobot SDK
# teleop 経由で record された Dex1 hand grasp point を再現するため。inference/ 側との
# drift は `tests/test_fk.py::TestConstantDriftFromInference` で cross-verify (test は
# self-contained 方針の例外)。
LEFT_WRIST_TOOL_OFFSET_M = np.array([0.087, -0.034, 0.092], dtype=np.float64)
RIGHT_WRIST_TOOL_OFFSET_M = np.array([0.116, -0.057, -0.015], dtype=np.float64)

# 19D layout (waist 3 + left_arm 7 + right_arm 7 + gripper 2) → SDK 29-joint index
# GR00T layout と同一 (物理 robot spec、shared truth)、gripper は wrist chain 不参加
_ACTION19_TO_SDK29_INDEX: tuple[int, ...] = (
    12, 13, 14,                        # waist 3
    15, 16, 17, 18, 19, 20, 21,        # left arm 7
    22, 23, 24, 25, 26, 27, 28,        # right arm 7
    -1, -1,                            # gripper 2 (wrist FK 不使用)
)

ACTION19_DIM: int = 19

# 19D layout 内の肘の index (FK の chain の途中で肘の位置を取り出す)
LEFT_ELBOW_ACTION19_INDEX: int = 6    # waist 3 + left shoulder 3
RIGHT_ELBOW_ACTION19_INDEX: int = 13  # waist 3 + left arm 7 + right shoulder 3

# `G1WristFKTorch.features` の 27 次元の並び (Issue #141 RO-16、FK の loss 用)。
# 位置は pelvis frame [m]、向きは wrist_yaw_link の回転行列の最初の 2 列 (rot6d)
FK_FEATURE_SLICES: dict[str, slice] = {
    "left_elbow": slice(0, 3),
    "left_hand": slice(3, 6),
    "left_rot6d": slice(6, 12),
    "right_elbow": slice(12, 15),
    "right_hand": slice(15, 18),
    "right_rot6d": slice(18, 24),
    "hand_diff": slice(24, 27),       # 両手先の位置の差 (左 − 右)
}
FK_FEATURE_DIM: int = 27


# ---------------------------------------------------------------------------
# Chain data structure
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# URDF parse (numpy 経路、init 時のみ実行、per-forward は不要)
# ---------------------------------------------------------------------------


def _euler_xyz_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler XYZ (radians) → 3x3 rotation matrix (Rz(yaw) @ Ry(pitch) @ Rx(roll))。"""
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


# ---------------------------------------------------------------------------
# Torch FK (per-forward、autograd 対応)
# ---------------------------------------------------------------------------


def _axis_angle_matrix_torch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula。axis (3,) unit + angle (...,) → R (..., 3, 3)、autograd 対応。"""
    norm = torch.linalg.norm(axis)
    if float(norm) < 1e-9:
        # 縮退 axis (回転無し扱い) → identity を batch shape で返す
        eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
        return eye.expand(*angle.shape, 3, 3).clone()
    a = axis / norm
    x, y, z = a[0], a[1], a[2]

    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1.0 - c

    # 各要素 shape (...,)、最後 stack で (..., 3, 3)
    row0 = torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], dim=-1)
    row1 = torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], dim=-1)
    row2 = torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _to_4x4(R: torch.Tensor) -> torch.Tensor:
    """R (..., 3, 3) → T (..., 4, 4) with zero translation, bottom row [0,0,0,1]。"""
    batch_shape = R.shape[:-2]
    T = torch.zeros(*batch_shape, 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., 3, 3] = 1.0
    return T


def _sdk29_to_action19_index(sdk_idx: int) -> int:
    """SDK 29-joint index → 19D action layout index の reverse lookup。

    Raises:
        ValueError: SDK idx が 19D 覆域外 (legs 0-11 は wrist chain 不参加なので通常起きない)。
    """
    for a_idx, s in enumerate(_ACTION19_TO_SDK29_INDEX):
        if s == sdk_idx:
            return a_idx
    raise ValueError(
        f"SDK joint index {sdk_idx} ({G1_JOINT_NAMES[sdk_idx]}) is not covered by "
        f"19D action layout (waist 3 + arms 14 + gripper 2)"
    )


class G1WristFKTorch(nn.Module):
    """19D action (waist 3 + arm 7 + arm 7 + gripper 2) → 両手 wrist 3D 位置
    (pelvis frame) の torch FK module。

    Init 時に URDF file を parse (numpy) → chain の fixed_T / axis を torch buffer に
    変換 + chain 各 step の 19D action index を precompute。per-forward は pure torch
    matmul chain、autograd で joint angle → 3D 位置の chain rule が伝播する。

    Buffer:
        _left_fixed_Ts:    (N_left, 4, 4) 各 chain step の parent → joint frame fixed transform
        _left_axes:        (N_left, 3)    各 chain step の joint rotation axis (unit)
        _left_action_idx:  (N_left,) long 各 chain step が使う 19D action index
        _left_tool_offset: (3,) wrist_yaw_link frame での tool point offset (Issue #134、
                           numpy `inference/desktop/perception/g1_urdf_fk.py::G1WristFK` と
                           意味論一致 = Dex1 grasp point)
        _right_*:          同上、右手 chain
    """

    def __init__(
        self,
        left_chain: Sequence[ChainJoint],
        right_chain: Sequence[ChainJoint],
        left_tool_offset: torch.Tensor | np.ndarray | None = None,
        right_tool_offset: torch.Tensor | np.ndarray | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        left_fixed_Ts, left_axes, left_action_idx = self._chain_to_tensors(left_chain, dtype)
        right_fixed_Ts, right_axes, right_action_idx = self._chain_to_tensors(right_chain, dtype)

        self.register_buffer("_left_fixed_Ts", left_fixed_Ts)
        self.register_buffer("_left_axes", left_axes)
        self.register_buffer("_left_action_idx", left_action_idx)
        self.register_buffer("_right_fixed_Ts", right_fixed_Ts)
        self.register_buffer("_right_axes", right_axes)
        self.register_buffer("_right_action_idx", right_action_idx)
        # 肘の joint が chain の何 step 目か (python int、forward で同期を起こさない)
        self._left_elbow_step = self._step_of(left_action_idx, LEFT_ELBOW_ACTION19_INDEX)
        self._right_elbow_step = self._step_of(right_action_idx, RIGHT_ELBOW_ACTION19_INDEX)

        # Issue #134: numpy `G1WristFK` と意味論を揃えるため tool_offset を適用。
        # None なら module default (`LEFT/RIGHT_WRIST_TOOL_OFFSET_M`) を tensor 化。
        # persistent=False: 定数なので checkpoint に保存しない (既 Phase K R-6 ckpt との
        # 互換維持、学習 param でも学習中更新される state でもない)。
        self.register_buffer(
            "_left_tool_offset",
            self._validate_offset(left_tool_offset, LEFT_WRIST_TOOL_OFFSET_M, "left_tool_offset", dtype),
            persistent=False,
        )
        self.register_buffer(
            "_right_tool_offset",
            self._validate_offset(right_tool_offset, RIGHT_WRIST_TOOL_OFFSET_M, "right_tool_offset", dtype),
            persistent=False,
        )

    @staticmethod
    def _validate_offset(
        value: torch.Tensor | np.ndarray | None,
        default: np.ndarray,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """None → default、値有りなら shape (3,) + finite verify して tensor 化。"""
        if value is None:
            return torch.tensor(default, dtype=dtype)
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone().to(dtype=dtype)
        else:
            tensor = torch.tensor(np.asarray(value), dtype=dtype)
        if tensor.shape != (3,):
            raise ValueError(f"{name} must be shape (3,), got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite, got {tensor.tolist()}")
        return tensor

    @staticmethod
    def _step_of(action_idx: torch.Tensor, target: int) -> int:
        hits = (action_idx == target).nonzero().flatten().tolist()
        if len(hits) != 1:
            raise ValueError(f"chain must contain 19D index {target} exactly once, got steps {hits}")
        return int(hits[0])

    @staticmethod
    def _chain_to_tensors(
        chain: Sequence[ChainJoint], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fixed_Ts = torch.stack(
            [torch.tensor(cj.fixed_T, dtype=dtype) for cj in chain], dim=0
        )
        axes = torch.stack([torch.tensor(cj.axis, dtype=dtype) for cj in chain], dim=0)
        action_idx = torch.tensor(
            [_sdk29_to_action19_index(cj.joint_index) for cj in chain], dtype=torch.long
        )
        return fixed_Ts, axes, action_idx

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path = DEFAULT_URDF_PATH,
        joint_names: Sequence[str] = G1_JOINT_NAMES,
        left_tool_offset: torch.Tensor | np.ndarray | None = None,
        right_tool_offset: torch.Tensor | np.ndarray | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "G1WristFKTorch":
        """URDF file を parse して instance を返す。default は model/ramen_ori/assets/urdf/。"""
        joint_name_to_index = {name: i for i, name in enumerate(joint_names)}
        left_chain = _parse_urdf_chain(urdf_path, LEFT_WRIST_LINK, joint_name_to_index)
        right_chain = _parse_urdf_chain(urdf_path, RIGHT_WRIST_LINK, joint_name_to_index)
        return cls(
            left_chain=left_chain,
            right_chain=right_chain,
            left_tool_offset=left_tool_offset,
            right_tool_offset=right_tool_offset,
            dtype=dtype,
        )

    @classmethod
    def from_default_urdf(
        cls,
        left_tool_offset: torch.Tensor | np.ndarray | None = None,
        right_tool_offset: torch.Tensor | np.ndarray | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "G1WristFKTorch":
        """model/ramen_ori/assets/urdf/g1_29dof_mode_15_with_dex1_1.urdf から init。"""
        return cls.from_urdf(
            urdf_path=DEFAULT_URDF_PATH,
            left_tool_offset=left_tool_offset,
            right_tool_offset=right_tool_offset,
            dtype=dtype,
        )

    def _fk_chain(
        self,
        q_action19: torch.Tensor,
        fixed_Ts: torch.Tensor,
        axes: torch.Tensor,
        action_idx: torch.Tensor,
        tool_offset: torch.Tensor,
        elbow_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """chain 経由で pelvis → target_link を計算し、肘の位置・tool point の位置・手の向きを返す (batch 対応)。

        Issue #134: `tool_offset` を wrist_yaw_link frame で加算 (numpy 側 `G1WristFK.
        compute_ee_state` と同じ order: `pos = origin + rot @ offset`)、意味論 = Dex1
        grasp point。
        Issue #141 RO-16: 同じ 1 回の chain の途中 (肘の joint frame の原点) で肘の位置を取り出す。

        Args:
            q_action19:  (..., 19) 19D action tensor
            fixed_Ts:    (N, 4, 4)
            axes:        (N, 3)
            action_idx:  (N,) long、各 chain step が使う 19D index
            tool_offset: (3,) wrist_yaw_link frame の tool point offset (buffer 経由)
            elbow_step:  肘の joint の chain step

        Returns:
            (elbow_pos (..., 3), tool_pos (..., 3), rot (..., 3, 3))、pelvis frame
        """
        # chain step 毎に該当 angle を index_select
        angles = torch.index_select(q_action19, dim=-1, index=action_idx)  # (..., N)

        batch_shape = q_action19.shape[:-1]
        # T の accumulator を identity で init
        T = torch.eye(4, dtype=q_action19.dtype, device=q_action19.device)
        T = T.expand(*batch_shape, 4, 4).clone() if batch_shape else T.clone()

        N = int(fixed_Ts.shape[0])
        elbow_pos = None
        for i in range(N):
            angle_i = angles[..., i]  # (...,)
            R_i = _axis_angle_matrix_torch(axes[i], angle_i)  # (..., 3, 3)
            T_joint = _to_4x4(R_i)  # (..., 4, 4)
            step_T = fixed_Ts[i] @ T_joint  # (4, 4) @ (..., 4, 4) → broadcast
            T = T @ step_T
            if i == elbow_step:
                # joint の回転は原点を動かさないので、この時点の原点 = 肘の joint の位置
                elbow_pos = T[..., :3, 3]

        # tool_offset apply: pos = origin + rot @ offset (numpy 側と同じ order)
        # rot (..., 3, 3) @ offset (3,) → (..., 3)、torch matmul は最後 2 dim を
        # matrix-vector product として扱う (numpy と同じ semantics)
        origin = T[..., :3, 3]                                        # (..., 3)
        rot = T[..., :3, :3]                                          # (..., 3, 3)
        tool_pos = origin + torch.matmul(rot, tool_offset.to(rot.dtype))  # (..., 3)
        return elbow_pos, tool_pos, rot

    def forward(self, q_action19: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """19D action → (left_wrist_pos, right_wrist_pos) 各 (..., 3)。

        Args:
            q_action19: (..., 19) torch tensor、任意 batch shape、requires_grad OK
                       Layout: waist(0:3) + left_arm(3:10) + right_arm(10:17) + gripper(17:19)

        Returns:
            (left_pos, right_pos): 各 (..., 3) tensor、pelvis frame 座標。
        """
        parts = self.forward_detailed(q_action19)
        return parts["left_hand"], parts["right_hand"]

    def forward_detailed(self, q_action19: torch.Tensor) -> dict[str, torch.Tensor]:
        """19D action → 左右の肘の位置・手先 (tool point) の位置・手の向き (Issue #141 RO-16)。

        bf16 の autocast の下でも FK 自身の精度 (buffer の dtype、既定 fp32) で計算する (Issue #141)。
        bf16 の回転行列の積では手先の位置が median 2 mm / 最大 6 mm ずれ、FK の loss と val の手首の誤差 (mm) に入る。

        Returns:
            {"left_elbow", "left_hand", "right_elbow", "right_hand"}: 各 (..., 3) pelvis frame 位置、
            {"left_rot", "right_rot"}: 各 (..., 3, 3) wrist_yaw_link の回転 (pelvis frame)
        """
        if q_action19.shape[-1] != ACTION19_DIM:
            raise ValueError(
                f"q_action19 last dim must be {ACTION19_DIM}, got {q_action19.shape}"
            )
        with torch.autocast(device_type=q_action19.device.type, enabled=False):
            q = q_action19.to(self._left_fixed_Ts.dtype)
            left_elbow, left_hand, left_rot = self._fk_chain(
                q,
                self._left_fixed_Ts,
                self._left_axes,
                self._left_action_idx,
                self._left_tool_offset,
                self._left_elbow_step,
            )
            right_elbow, right_hand, right_rot = self._fk_chain(
                q,
                self._right_fixed_Ts,
                self._right_axes,
                self._right_action_idx,
                self._right_tool_offset,
                self._right_elbow_step,
            )
        return {
            "left_elbow": left_elbow, "left_hand": left_hand, "left_rot": left_rot,
            "right_elbow": right_elbow, "right_hand": right_hand, "right_rot": right_rot,
        }

    def features(self, q_action19: torch.Tensor) -> torch.Tensor:
        """19D action → FK の loss 用の 27 次元 (並びは FK_FEATURE_SLICES)。

        手の向きは回転行列の最初の 2 列 (rot6d、連続で ±π の飛びが無い)。
        """
        p = self.forward_detailed(q_action19)

        def _rot6d(rot: torch.Tensor) -> torch.Tensor:
            return torch.cat([rot[..., :, 0], rot[..., :, 1]], dim=-1)

        return torch.cat(
            [
                p["left_elbow"], p["left_hand"], _rot6d(p["left_rot"]),
                p["right_elbow"], p["right_hand"], _rot6d(p["right_rot"]),
                p["left_hand"] - p["right_hand"],
            ],
            dim=-1,
        )


def assemble_action19(
    action_waist_3: torch.Tensor,
    action_arms_hand_16: torch.Tensor,
) -> torch.Tensor:
    """RAMEN-Ori の 16D action + teacher waist 3D → 19D layout に concat。

    Args:
        action_waist_3: (..., 3) teacher の waist target (data_lerobot が
                        action.robot_q_desired[:, waist_indices] から供給)
        action_arms_hand_16: (..., 16) pred or teacher の arms 14 + hand 2

    Returns:
        (..., 19) tensor、layout = waist(0:3) + arms_hand(3:19)。
        RAMEN-Ori 16D の内部 layout (arms 14 + hand 2) は GR00T layout
        (arm 7 + arm 7 + gripper 2) と一致するため、単純 concat で 19D 完成。
    """
    if action_waist_3.shape[-1] != 3:
        raise ValueError(f"action_waist_3 last dim must be 3, got {action_waist_3.shape}")
    if action_arms_hand_16.shape[-1] != 16:
        raise ValueError(
            f"action_arms_hand_16 last dim must be 16, got {action_arms_hand_16.shape}"
        )
    return torch.cat([action_waist_3, action_arms_hand_16], dim=-1)
