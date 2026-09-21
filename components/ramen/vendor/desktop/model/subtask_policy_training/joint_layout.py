"""GR00T action / state 19-dim layout の単一 source-of-truth (Issue #129 Phase 0 D0-4)。

GR00T の action / state は独自 19-dim layout で、SDK 29-joint order (`G1_JOINT_NAMES`
in `inference/desktop/perception/g1_urdf_fk.py`) とは順序が違う。T0-1 (左手 loss weight)、
L1 (bilateral)、L4 (FK anchor) の実装で「左腕とは何 index か / 左右対称は何 index か」
を毎回 hardcode するのを避けるため、この module に定数化してすべての Phase から参照する。

# 参照元

- Names: `configs/subtask_training.json` の "state".names + "action".names (19 dim、同一 layout)
- Mapping to SDK 29-joint: `inference/desktop/perception/g1_urdf_fk.py:G1_JOINT_NAMES`
"""

from __future__ import annotations


# GR00T action / state layout (19 dim、subtask_training.json と一致)。
GROOT_ACTION_DIM: int = 19

GROOT_ACTION_NAMES: tuple[str, ...] = (
    "waist_yaw_joint",             # 0
    "waist_roll_joint",            # 1
    "waist_pitch_joint",           # 2
    "left_shoulder_pitch_joint",   # 3
    "left_shoulder_roll_joint",    # 4
    "left_shoulder_yaw_joint",     # 5
    "left_elbow_joint",            # 6
    "left_wrist_roll_joint",       # 7
    "left_wrist_pitch_joint",      # 8
    "left_wrist_yaw_joint",        # 9
    "right_shoulder_pitch_joint",  # 10
    "right_shoulder_roll_joint",   # 11
    "right_shoulder_yaw_joint",    # 12
    "right_elbow_joint",           # 13
    "right_wrist_roll_joint",      # 14
    "right_wrist_pitch_joint",     # 15
    "right_wrist_yaw_joint",       # 16
    "left_gripper_q",              # 17
    "right_gripper_q",             # 18
)


# 部位別 index (GR00T 19-dim 空間)
WAIST_INDICES: tuple[int, ...] = (0, 1, 2)
LEFT_ARM_INDICES: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9)
RIGHT_ARM_INDICES: tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16)
LEFT_GRIP_INDEX: int = 17
RIGHT_GRIP_INDEX: int = 18

# T0-1 (左手 loss weight up 対象) = 左腕 7 + 左 grip 1 = 8 dim
LEFT_HAND_INDICES: tuple[int, ...] = LEFT_ARM_INDICES + (LEFT_GRIP_INDEX,)
RIGHT_HAND_INDICES: tuple[int, ...] = RIGHT_ARM_INDICES + (RIGHT_GRIP_INDEX,)

# L1 bilateral 対象 = 左右腕 7 dim ずつ (grip は左右で意味論異なる可能性ありまず arm のみ)
BILATERAL_LEFT_INDICES: tuple[int, ...] = LEFT_ARM_INDICES
BILATERAL_RIGHT_INDICES: tuple[int, ...] = RIGHT_ARM_INDICES


# GR00T 19-dim → SDK 29-joint index の mapping (L4 FK anchor 用)。
# GR00T[i] → SDK[GROOT_TO_SDK29_INDEX[i]] の joint に相当。
# gripper (17, 18) は SDK 29-joint に含まれない (hand_cmd 別系統) のため -1 sentinel。
GROOT_TO_SDK29_INDEX: tuple[int, ...] = (
    12,  # waist_yaw
    13,  # waist_roll
    14,  # waist_pitch
    15,  # left_shoulder_pitch
    16,  # left_shoulder_roll
    17,  # left_shoulder_yaw
    18,  # left_elbow
    19,  # left_wrist_roll
    20,  # left_wrist_pitch
    21,  # left_wrist_yaw
    22,  # right_shoulder_pitch
    23,  # right_shoulder_roll
    24,  # right_shoulder_yaw
    25,  # right_elbow
    26,  # right_wrist_roll
    27,  # right_wrist_pitch
    28,  # right_wrist_yaw
    -1,  # left_gripper_q (SDK 29 外、hand_cmd[0])
    -1,  # right_gripper_q (SDK 29 外、hand_cmd[1])
)


def validate() -> None:
    """自己一貫性 check (import 時に呼ばれる想定なし、test 用)。"""
    assert len(GROOT_ACTION_NAMES) == GROOT_ACTION_DIM
    assert len(GROOT_TO_SDK29_INDEX) == GROOT_ACTION_DIM
    assert len(WAIST_INDICES) == 3
    assert len(LEFT_ARM_INDICES) == 7
    assert len(RIGHT_ARM_INDICES) == 7
    assert len(LEFT_HAND_INDICES) == 8
    assert len(RIGHT_HAND_INDICES) == 8

    # 重複なし
    all_body = tuple(WAIST_INDICES) + tuple(LEFT_ARM_INDICES) + tuple(RIGHT_ARM_INDICES)
    assert len(set(all_body)) == 17, f"body indices duplicated: {all_body}"
    assert set(all_body) | {LEFT_GRIP_INDEX, RIGHT_GRIP_INDEX} == set(range(GROOT_ACTION_DIM))

    # gripper は SDK 29 外
    assert GROOT_TO_SDK29_INDEX[LEFT_GRIP_INDEX] == -1
    assert GROOT_TO_SDK29_INDEX[RIGHT_GRIP_INDEX] == -1
    # body 部分は 0..28 の範囲、重複なし
    body_sdk = [GROOT_TO_SDK29_INDEX[i] for i in all_body]
    assert all(0 <= x < 29 for x in body_sdk)
    assert len(set(body_sdk)) == 17
