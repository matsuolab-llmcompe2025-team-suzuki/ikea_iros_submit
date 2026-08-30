"""GR00T 38D action → decoupled boundary ``(T, 25)`` task-space adapter。

VENDORED from iros_2026_ramen
`inference/desktop/lower_policy/policies/taskspace_adapter.py`
(self-contained Thor image のため submission repo に複製)。pure numpy。

GHCR boundary (`ikea_iros_submit`) の `decoupled` lane は policy に
``(T, 25)`` task-space chunk を要求する (row layout は下記 / `boundary/actions.py`)。
本 module は GR00T pick_legs 系の raw action (robot_q_desired 36 + hand_cmd 2 = 38D、
ABSOLUTE, new_embodiment) を、その ``(T, 25)`` に変換する **pure numpy** adapter。

FK は既存 `perception/g1_urdf_fk.G1WristFK`(pelvis frame + 5cm tool offset)を
再利用する。EE 姿勢は euler を経由せず回転行列→quat 直変換する。

``(T, 25)`` row layout (`boundary/actions.py` と一致、並べ替え禁止):
    [0:2]   左手 2 指  (-1=open, +1=closed)
    [2:4]   右手 2 指  (同)
    [4:7]   左 EE 位置 xyz [m]
    [7:11]  左 EE quat (w, x, y, z)
    [11:14] 右 EE 位置
    [14:18] 右 EE quat (w, x, y, z)
    [18:21] navigate_cmd (vx, vy, yaw_rate)
    [21]    base_height_cmd
    [22:25] torso_orientation_rpy_cmd (roll, pitch, yaw)

**Frame は後付け吸収可能**: organizer decoupled IK が期待する EE 基準 frame が
未確定 (調査 Q1) のため、`ee_frame_transform` (4x4) で pelvis→期待 frame の
変換を後から差せる。default = None (pelvis frame、GR00T 訓練時と同一)。
下半身 (navigate / base_height / torso_rpy) は pick 系 (lower_body=0) 想定で
default 0-hold。waist を動かす skill は呼出側で明示指定する。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# GR00T new_embodiment action の内部構造 (groot_pick_leg_contract 準拠)。
_ROBOT_Q_DIM = 36          # root(7) + body(29)
_ROOT_DIM = 7              # xyz(3) + wxyz(4)
_BODY_DIM = 29             # legs(12) + waist(3) + Larm(7) + Rarm(7)
_HAND_DIM = 2              # Dex1 opening (left, right)
GROOT_ACTION_DIM = _ROBOT_Q_DIM + _HAND_DIM   # 38
TASKSPACE_DIM = 25

# Dex1 model 空間の開き上限 (groot_pick_leg_contract.DEX1_DATASET_OPEN_VALUE)。
# model 値 = 開き量 [rad 相当]、DEX1_OPEN_VALUE = 全開。boundary は -1=open / +1=closed。
DEX1_OPEN_VALUE: float = 4.5


def rotation_matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """3x3 回転行列 → 単位 quaternion (w, x, y, z)。

    Shepperd 法 (数値安定、gimbal lock 無し)。返り値は w>=0 に正規化。
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"matrix must be (3, 3), got {m.shape}")
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:   # canonical hemisphere (w >= 0)
        q = -q
    return q


def dex1_model_to_taskspace(hand_value: float) -> float:
    """Dex1 model 空間の開き量 → boundary 手指値 [-1(open), +1(closed)]。"""
    frac_open = float(np.clip(hand_value / DEX1_OPEN_VALUE, 0.0, 1.0))
    return 1.0 - 2.0 * frac_open


def _apply_frame(
    pos: np.ndarray, rot: np.ndarray, transform: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """pelvis frame の (pos, R) を transform (4x4) で別 frame へ写す。"""
    if transform is None:
        return pos, rot
    tf = np.asarray(transform, dtype=np.float64)
    if tf.shape != (4, 4):
        raise ValueError(f"ee_frame_transform must be (4, 4), got {tf.shape}")
    rf = tf[:3, :3]
    return rf @ pos + tf[:3, 3], rf @ rot


def groot_action_to_taskspace(
    action_38: np.ndarray,
    fk,
    *,
    navigate_cmd: Sequence[float] = (0.0, 0.0, 0.0),
    base_height_cmd: float = 0.0,
    torso_rpy_cmd: Sequence[float] = (0.0, 0.0, 0.0),
    ee_frame_transform: np.ndarray | None = None,
) -> np.ndarray:
    """GR00T raw action 1 row (38,) → boundary task-space (25,)。

    Args:
        action_38: (38,) = robot_q_desired(36: root7 + body29) + hand_cmd(2)。ABSOLUTE。
        fk: `G1WristFK` instance (compute_ee_transforms を持つ)。
        navigate_cmd / base_height_cmd / torso_rpy_cmd: 下半身 command。pick 系は
            default 0-hold。waist を動かす skill は呼出側で指定する。
        ee_frame_transform: pelvis→期待 EE frame の 4x4 変換 (Q1 未確定用の後付け穴)。
            None = pelvis frame そのまま。

    Returns:
        (25,) float32、`boundary/actions.py` DecoupledSink.validate_chunk 準拠。
    """
    a = np.asarray(action_38, dtype=np.float64)
    if a.shape != (GROOT_ACTION_DIM,):
        raise ValueError(
            f"action must be ({GROOT_ACTION_DIM},) = robot_q(36)+hand(2), got {a.shape}"
        )
    body29 = a[_ROOT_DIM:_ROBOT_Q_DIM]          # root(7) を捨てて body(29)
    hand = a[_ROBOT_Q_DIM:GROOT_ACTION_DIM]     # (2,) left, right

    left_pos, left_R, right_pos, right_R = fk.compute_ee_transforms(body29)
    left_pos, left_R = _apply_frame(left_pos, left_R, ee_frame_transform)
    right_pos, right_R = _apply_frame(right_pos, right_R, ee_frame_transform)

    out = np.zeros(TASKSPACE_DIM, dtype=np.float32)
    out[0:2] = dex1_model_to_taskspace(float(hand[0]))   # 左手 2 指に同値
    out[2:4] = dex1_model_to_taskspace(float(hand[1]))   # 右手 2 指に同値
    out[4:7] = left_pos
    out[7:11] = rotation_matrix_to_quat_wxyz(left_R)
    out[11:14] = right_pos
    out[14:18] = rotation_matrix_to_quat_wxyz(right_R)
    out[18:21] = np.asarray(navigate_cmd, dtype=np.float32)
    out[21] = float(base_height_cmd)
    out[22:25] = np.asarray(torso_rpy_cmd, dtype=np.float32)
    return out


def groot_chunk_to_taskspace(
    chunk_38: np.ndarray, fk, **kwargs
) -> np.ndarray:
    """GR00T raw action chunk (T, 38) → boundary (T, 25)。

    kwargs は `groot_action_to_taskspace` に横流し (全 row 共通適用)。
    """
    chunk = np.asarray(chunk_38, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != GROOT_ACTION_DIM:
        raise ValueError(
            f"chunk must be (T, {GROOT_ACTION_DIM}), got {chunk.shape}"
        )
    rows = [groot_action_to_taskspace(row, fk, **kwargs) for row in chunk]
    return np.stack(rows).astype(np.float32, copy=False)
