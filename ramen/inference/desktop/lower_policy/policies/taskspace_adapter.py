"""GR00T 38D action → decoupled boundary ``(T, 25)`` task-space adapter。

GHCR boundary (`ikea_iros_submit`) の `decoupled` lane は policy に
``(T, 25)`` task-space chunk を要求する (row layout は下記 / `boundary/actions.py`)。
本 module は GR00T pick_legs 系の raw action (robot_q_desired 36 + hand_cmd 2 = 38D、
ABSOLUTE, new_embodiment) を、その ``(T, 25)`` に変換する **pure numpy** adapter。

FK は既存 `perception/g1_urdf_fk.G1WristFK` の zero-tool-offset
`wrist_yaw_link` poseを使う。EE 姿勢は euler を経由せず回転行列→quat 直変換する。

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
下半身 (navigate / torso_rpy) は pick 系 (lower_body=0) 想定で default 0-hold。
waist を動かす skill は呼出側で明示指定する。**base_height だけは 0 が
「0 m へ沈め」の意味になる**ので既定は中立 0.74 m (`DEFAULT_BASE_HEIGHT_M`)。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# GR00T new_embodiment action の内部構造 (groot_pick_leg_contract 準拠)。
_ROBOT_Q_DIM = 36  # root(7) + body(29)
_ROOT_DIM = 7  # xyz(3) + wxyz(4)
_BODY_DIM = 29  # legs(12) + waist(3) + Larm(7) + Rarm(7)
_HAND_DIM = 2  # Dex1 opening (left, right)
GROOT_ACTION_DIM = _ROBOT_Q_DIM + _HAND_DIM  # 38
TASKSPACE_DIM = 25

# Dex1 model 空間の開き上限 (groot_pick_leg_contract.DEX1_DATASET_OPEN_VALUE)。
# model 値 = Dex1 モータの**物理角 [rad]** (0=閉)。学習データの上限が 4.5 で、
# ハードウェアの全開 (5.4) ではない。
DEX1_OPEN_VALUE: float = 4.5

# 運営 relay の全開 [rad]。`tools/run_wbc_with_dex1.py` の `hand_norm_to_dex1_q` は
# boundary の -1 (open) .. +1 (closed) を q = -5.30 .. 0 へ**線形に**写す
# (符号は Unitree の reference と逆で、大きさは同じ物理角)。
#
# ⚠️ **model の 4.5 を boundary の -1 に写してはいけない** (Issue #159 B3a-04)。
# 以前はそうしていて、会場では物理角が 5.30/4.5 = 1.18 倍になっていた。
# 脚を握るとき model は 1.85 を出し、脚に当たって 2.15 で止まる (ラボ実測、
# 締め付けは差の 0.3 rad)。1.18 倍すると指令が 2.18 = **脚より外側**になり、
# 締め付けがほぼ消える。物理角をそのまま渡す。
DEX1_BOUNDARY_FULL_OPEN_RAD: float = 5.30

# 骨盤高さ目標 [m]。**0.0 は「指定なし」ではない。**
#
# 運営の adapter は col [21] を WBC へ**リテラル転送**する:
#     reference/wbc_adapter/wbc_driver.py の run_decoupled
#         base_height_command=[[float(r[21])] for r in rows[:n]]
# 既定値は運営側にも同じ名前で存在し、NVIDIA の
# `decoupled_wbc/control/main/constants.py` 由来:
#     reference/wbc_adapter/wbc_goal.py:9,46
#         DEFAULT_BASE_HEIGHT = 0.74   # matches the (T,25) col [21] convention
#
# つまり 0.0 を入れると「骨盤を高さ 0 m へ」= 床まで沈む指令になる。go-live の
# 1 通目から効くので、呼出側が何も渡さないときは中立姿勢を出す。
# (運営 package の reference/wbc_adapter/wbc_driver.py run_decoupled がリテラル転送)
DEFAULT_BASE_HEIGHT_M: float = 0.74

# 骨盤高さとして通してよい範囲 [m]。**歯止めであって clamp ではない。**
#
# col [21] は運営側でも `boundary/actions.py` でも範囲を検証されない
# (hand は [-1,1]、quat は unit norm を見るのに、下半身列だけ素通し)。
# 実際 `0.0` が数週間どこにも引っかからずに WBC へ届いていた。
# 単位違いや config の typo をここで止める。
BASE_HEIGHT_MIN_M: float = 0.3
BASE_HEIGHT_MAX_M: float = 1.0


def check_base_height(base_height_cmd: float) -> float:
    """骨盤高さ目標が妥当な範囲かを見る。範囲外なら `ValueError`。

    clamp しないのは、**黙って直すと「なぜ違う値が出たか」が分からなくなる**ため。
    呼出側 (driver / run_live) は例外を HOLD に変換するので、fail-safe に倒れる。
    """
    value = float(base_height_cmd)
    if not BASE_HEIGHT_MIN_M <= value <= BASE_HEIGHT_MAX_M:
        raise ValueError(
            f"base_height_cmd {value} is outside the plausible pelvis height "
            f"range [{BASE_HEIGHT_MIN_M}, {BASE_HEIGHT_MAX_M}] m. "
            "0 means 'sink the pelvis to the floor', not 'leave it unset' — "
            "the organizer relays col [21] literally (wbc_adapter/wbc_driver.py run_decoupled)."
        )
    return value


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
    if q[0] < 0.0:  # canonical hemisphere (w >= 0)
        q = -q
    return q


def dex1_model_to_taskspace(hand_value: float) -> float:
    """Dex1 model 空間の開き量 → boundary 手指値 [-1(open), +1(closed)]。

    運営 relay を通った後の物理角が入力値と一致するように写す
    (model 4.5 → boundary -0.698 → 運営 q -4.5)。policy出力は上流の
    teacher-rangeで学習範囲に制限される一方、安全な準備・終了手順は5.30radの
    全開を要求するため、ここでは物理範囲 [0, 5.30] にだけclipする。
    """
    opening = float(np.clip(hand_value, 0.0, DEX1_BOUNDARY_FULL_OPEN_RAD))
    return 1.0 - 2.0 * opening / DEX1_BOUNDARY_FULL_OPEN_RAD


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
    base_height_cmd: float = DEFAULT_BASE_HEIGHT_M,
    torso_rpy_cmd: Sequence[float] = (0.0, 0.0, 0.0),
    ee_frame_transform: np.ndarray | None = None,
) -> np.ndarray:
    """GR00T raw action 1 row (38,) → boundary task-space (25,)。

    Args:
        action_38: (38,) = robot_q_desired(36: root7 + body29) + hand_cmd(2)。ABSOLUTE。
        fk: `G1WristFK` instance (compute_ee_transforms を持つ)。
        navigate_cmd / torso_rpy_cmd: 下半身 command。pick 系は default 0-hold。
            waist を動かす skill は呼出側で指定する。
        base_height_cmd: 骨盤高さ目標 [m]。**0 は「指定なし」ではなく 0 m 指令**
            なので既定は中立の `DEFAULT_BASE_HEIGHT_M`。
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
    body29 = a[_ROOT_DIM:_ROBOT_Q_DIM]  # root(7) を捨てて body(29)
    hand = a[_ROBOT_Q_DIM:GROOT_ACTION_DIM]  # (2,) left, right

    left_pos, left_R, right_pos, right_R = fk.compute_ee_transforms(body29)
    left_pos, left_R = _apply_frame(left_pos, left_R, ee_frame_transform)
    right_pos, right_R = _apply_frame(right_pos, right_R, ee_frame_transform)

    out = np.zeros(TASKSPACE_DIM, dtype=np.float32)
    out[0:2] = dex1_model_to_taskspace(float(hand[0]))  # 左手 2 指に同値
    out[2:4] = dex1_model_to_taskspace(float(hand[1]))  # 右手 2 指に同値
    out[4:7] = left_pos
    out[7:11] = rotation_matrix_to_quat_wxyz(left_R)
    out[11:14] = right_pos
    out[14:18] = rotation_matrix_to_quat_wxyz(right_R)
    out[18:21] = np.asarray(navigate_cmd, dtype=np.float32)
    out[21] = check_base_height(base_height_cmd)
    out[22:25] = np.asarray(torso_rpy_cmd, dtype=np.float32)
    return out


def groot_chunk_to_taskspace(chunk_38: np.ndarray, fk, **kwargs) -> np.ndarray:
    """GR00T raw action chunk (T, 38) → boundary (T, 25)。

    kwargs は `groot_action_to_taskspace` に横流し (全 row 共通適用)。
    """
    chunk = np.asarray(chunk_38, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != GROOT_ACTION_DIM:
        raise ValueError(f"chunk must be (T, {GROOT_ACTION_DIM}), got {chunk.shape}")
    rows = [groot_action_to_taskspace(row, fk, **kwargs) for row in chunk]
    return np.stack(rows).astype(np.float32, copy=False)
