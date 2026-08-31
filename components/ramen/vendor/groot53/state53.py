"""53D REAL_G1_RELATIVE_EEF の 49D state 組立 + 53D→19D slice (pure numpy)。

VENDORED/抽出 from iros_2026_ramen inference/desktop/lower_policy/policies/groot.py
(build_state_from_raw / _wrist_pose_to_xyz_rot6d / _euler_xyz_to_rot6d /
slice_53d_to_19d) + model/subtask_policy_training/gr00t/dex1_hand_synergy。
訓練仕様の厳密追従が要るので、上流変更時は本 copy も同期すること。
"""

from __future__ import annotations

import numpy as np

from dex1.dex1_hand_synergy import dex1_to_hand, hand_to_dex1

STATE_DIM = 49
ACTION_DIM = 53
DEX1_OPEN_VALUE = 4.5

# G1_JOINT_NAMES (29-DoF) の関節 slice。
_G1_WAIST = slice(12, 15)
_G1_LEFT_ARM = slice(15, 22)
_G1_RIGHT_ARM = slice(22, 29)

# 53D REAL_G1_RELATIVE_EEF action slice (g1_full_body_mapping 準拠)。
_A53_LEFT_HAND = slice(18, 25)
_A53_RIGHT_HAND = slice(25, 32)
_A53_LEFT_ARM = slice(32, 39)
_A53_RIGHT_ARM = slice(39, 46)
_A53_WAIST = slice(46, 49)


def _euler_xyz_to_rot6d(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler XYZ → ROT6D (Rz(yaw)@Ry(pitch)@Rx(roll) の先頭2行、flatten)。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    row0 = np.array(
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dtype=np.float32
    )
    row1 = np.array(
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dtype=np.float32
    )
    return np.concatenate([row0, row1])


def _wrist_pose_to_xyz_rot6d(pose6d: np.ndarray) -> np.ndarray:
    """(xyz + euler xyz) 6D → (xyz + rot6d) 9D。"""
    xyz = np.asarray(pose6d[:3], dtype=np.float32)
    rot6d = _euler_xyz_to_rot6d(float(pose6d[3]), float(pose6d[4]), float(pose6d[5]))
    return np.concatenate([xyz, rot6d])


def build_state_49d(
    body_q29: np.ndarray, dex1_open_fraction, ee_state12: np.ndarray
) -> np.ndarray:
    """boundary body_q(29) + Dex1 開度 + ee_state(12) → GR00T 49D state。

    ee_state12 = G1WristFK.compute_ee_state 出力 = 左(xyz+euler)+右(xyz+euler)。
    dex1_open_fraction (2,) ∈ [0,1] → physical Dex1 [rad] = ×DEX1_OPEN_VALUE。

    Layout: [0:9] Lwrist_eef_9d [9:18] Rwrist_eef_9d [18:25] Lhand synergy
            [25:32] Rhand synergy [32:39] Larm [39:46] Rarm [46:49] waist。
    """
    body_q = np.asarray(body_q29, dtype=np.float32)
    if body_q.shape != (29,):
        raise ValueError(f"body_q must be (29,), got {body_q.shape}")
    ee = np.asarray(ee_state12, dtype=np.float32)
    if ee.shape != (12,):
        raise ValueError(f"ee_state must be (12,), got {ee.shape}")
    hand_rad = np.clip(np.asarray(dex1_open_fraction, np.float32), 0.0, 1.0) * DEX1_OPEN_VALUE

    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[0:9] = _wrist_pose_to_xyz_rot6d(ee[0:6])
    out[9:18] = _wrist_pose_to_xyz_rot6d(ee[6:12])
    out[18:25] = dex1_to_hand(float(hand_rad[0]), side="left", kind="state")
    out[25:32] = dex1_to_hand(float(hand_rad[1]), side="right", kind="state")
    out[32:39] = body_q[_G1_LEFT_ARM]
    out[39:46] = body_q[_G1_RIGHT_ARM]
    out[46:49] = body_q[_G1_WAIST]
    return out


def slice_53d_to_19d(action_chunk_53d: np.ndarray) -> np.ndarray:
    """(T,53) decoded action → (T,19) = waist3 + Larm7 + Rarm7 + Ldex1 + Rdex1。"""
    a = np.asarray(action_chunk_53d)
    if a.ndim != 2 or a.shape[1] != ACTION_DIM:
        raise ValueError(f"action must be (T,{ACTION_DIM}), got {a.shape}")
    t = a.shape[0]
    out = np.zeros((t, 19), dtype=a.dtype)
    out[:, 0:3] = a[:, _A53_WAIST]
    out[:, 3:10] = a[:, _A53_LEFT_ARM]
    out[:, 10:17] = a[:, _A53_RIGHT_ARM]
    for i in range(t):
        out[i, 17] = hand_to_dex1(a[i, _A53_LEFT_HAND].tolist(), side="left", kind="action")
        out[i, 18] = hand_to_dex1(a[i, _A53_RIGHT_HAND].tolist(), side="right", kind="action")
    return out
