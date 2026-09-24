"""RAMEN-Ori state derive (Issue #120 Phase A-1 で E-δ、Alt-2 で E-β 追加)。

source LeRobot v3 の 1 frame (obs 38D + 1 frame 前の action 38D + ee_state 12D + prev obs 38D) から
design doc §5 の state を組む。suzuki の GR00T mapping
(`model/subtask_policy_training/gr00t/g1_full_body_mapping.py::UPPER_BODY_SOURCE_INDEX_MAP`)
を再利用して joint 19 の index 選択を一元化する。

# E-δ 71D 内訳 (design doc §5)

| slice | dim | source |
|---|---:|---|
| [0:19] joint | 19 | q_current[UPPER_BODY] (waist3+arm7+arm7+hand2) |
| [19:38] tracking_err | 19 | q_desired[t-1][UPPER_BODY] - q_current[t][UPPER_BODY]、腰 3 は常に 0、区間先頭は zeros |
| [38:57] velocity | 19 | (q_current[t] - q_current[t-1])[UPPER_BODY], ep 先頭 は zeros |
| [57:59] hand_state | 2 | q_current[36:38] = hand_state raw (joint 末尾 2 と重複、design doc の explicit token) |
| [59:71] ee_pose | 12 | ee_state[:12] (left+right xyz+euler xyz) |

tracking_err は推論 (`build_state_from_raw`) の「前 tick に送った指令 − 今」と同じ定義
(Issue #141 RO-3)。今の frame の指令 q_desired[t] は正解 action の row 0 そのものなので入力に使わない。
腰は model が出力しない (action は arms 14 + hand 2) ため、推論の「前の指令」に腰の値が無く、学習・推論とも 0。

# E-β 73D 内訳 (Alt-2)

E-δ 71D + [71:73] wrist depth contact bit (2D、left/right)。design doc §5 E-β。
depth contact bit は wrist RGB→SGBM+WLS depth の 「wrist から手前 物体までの距離 <
threshold = 接触」ソフト signal (contact-rich task の力覚代替)。SGBM pipeline は
Phase 0 で precompute (別 Alt)、Alt-2 時点では placeholder zero で API 経路のみ確立。

# API

`derive_state_71d` / `derive_state_73d` は pure numpy 関数、LeRobot key への依存を
Dataset 層に閉じ込める。
"""

from __future__ import annotations

import numpy as np

from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    SOURCE_EEF_DIM,
    SOURCE_ROBOT_Q_DIM,
    SOURCE_STATE_DIM,
    UPPER_BODY_SOURCE_INDEX_MAP,
    UPPER_BODY_STATE_DIM,
)


RAMEN_ORI_STATE_DIM = 71                        # E-δ default
RAMEN_ORI_STATE_DIM_WITH_DEPTH_CONTACT = 73     # E-β (+ depth contact bit 2D)
DEPTH_CONTACT_DIM = 2                            # left / right wrist

# Issue #129 Phase A (2026-08-31): action target は arms 14 + hand 2 = 16D
# (waist 3 除外)。inference で waist を使わない (別 issue で確定) ため、
# loss 圏外に外して arms/hand への gradient 集中を狙う。
# UPPER_BODY_SOURCE_INDEX_MAP は waist(3)+left_arm(7)+right_arm(7)+hand(2)=19、
# その先頭 3 dim を dropped した 16 index を新たに定義する。
RAMEN_ORI_ACTION_DIM = 16
ARMS_HAND_SOURCE_INDEX_MAP = tuple(UPPER_BODY_SOURCE_INDEX_MAP[3:])   # arms 14 + hand 2

# Issue #129 Phase F (2026-08-31): 16D action 内の layout。
# Relative action space (Run 3/5) では arms 14 dim を Δq に、hand 2 dim は absolute。
# GR00T の relative_exclude_joints=[hand,...] と同じ思想 (hand は discrete state 崩さない)。
ACTION16_ARMS_SLICE = slice(0, 14)   # arms 14 (left arm 7 + right arm 7)
ACTION16_HAND_SLICE = slice(14, 16)  # hand 2 (left grip + right grip)

# 71D state 内の arms 14 dim slice (E-δ layout: joint 19 = waist 3 + arms 14 + hand 2)。
# Current q を Δq 積分の初期値として取り出す時に使う (Phase F L4 relative 対応)。
STATE71_ARMS_SLICE = slice(3, 17)    # state[3:17] = 14 dim arms current joint
STATE71_HAND_SLICE = slice(17, 19)   # state[17:19] = 2 dim hand current

# 71D state の中身ごとの区切り (module docstring の表と同じ)。正規化の log などで使う
STATE71_GROUPS: dict[str, slice] = {
    "joint": slice(0, 19),
    "tracking_err": slice(19, 38),
    "velocity": slice(38, 57),
    "hand_state": slice(57, 59),
    "ee_pose": slice(59, 71),
}

# 学習時に state を group ごとに隠す変種 (Issue #141 Phase 6、StateEncoder の dropout) の group。並びが隠した印の並び。
# hand の値は joint の末尾 2 と hand_state に同じ値で入っているので、hand の group は両方を隠す (joint は腰 3 + 腕 14)
STATE71_DROPOUT_GROUPS: dict[str, tuple[slice, ...]] = {
    "joint": (slice(0, 17),),
    "hand": (slice(17, 19), slice(57, 59)),
    "tracking_err": (slice(19, 38),),
    "velocity": (slice(38, 57),),
    "ee_pose": (slice(59, 71),),
}

# ckpt の約束 (contract.py) に書く 71D state の定義。定義を変えたら version を上げる。
# version 1 (Phase K の ckpt、約束なし): tracking_err = 今の frame の指令 − 今 (正解 action の row 0 が入る)、腰も含む
STATE71_DEFINITION: dict = {
    "version": 2,
    "groups": {name: [sl.start, sl.stop] for name, sl in STATE71_GROUPS.items()},
    "tracking_err": "1 frame 前の指令 − 今。腰の 3 次元は 0、区間の先頭 (推論の 1 tick 目) は 0",
    "velocity": "今 − 1 frame 前の関節角。区間の先頭は 0",
}

# Issue #129 Phase B (2026-08-31): L4 FK anchor loss で必要な teacher waist の source index。
# UPPER_BODY_SOURCE_INDEX_MAP の先頭 3 dim = waist、data_lerobot.py で
# action.robot_q_desired[:, WAIST_SOURCE_INDEX_MAP] を切り出して batch["action_waist_teacher"]
# として供給、fk.py の assemble_action19 で pred_arms_hand と concat 済 19D にして FK。
WAIST_SOURCE_INDEX_MAP = tuple(UPPER_BODY_SOURCE_INDEX_MAP[:3])       # waist 3

_UPPER_INDEX = np.asarray(UPPER_BODY_SOURCE_INDEX_MAP, dtype=np.int64)
_ARMS_HAND_INDEX = np.asarray(ARMS_HAND_SOURCE_INDEX_MAP, dtype=np.int64)
_WAIST_INDEX = np.asarray(WAIST_SOURCE_INDEX_MAP, dtype=np.int64)

# 19D (UPPER_BODY) 内の腰 3 次元。tracking_err では常に 0 にする。
UPPER19_WAIST_SLICE = slice(0, 3)


def derive_state_71d(
    state_current: np.ndarray,
    action_prev: np.ndarray | None,
    ee_state: np.ndarray,
    state_prev: np.ndarray | None,
) -> np.ndarray:
    """1 frame 分の 71D E-δ state を組む。

    Args:
        state_current: (38,) source `observation.state` = q_current(36) + hand_state(2)
        action_prev: (38,) 1 frame 前の source `action` = q_desired(36) + hand_cmd(2)。
            区間先頭 (1 frame 前が区間の外) は None (tracking_err=zeros、推論の 1 tick 目と同じ)
        ee_state: (12,) source EE pose (left+right の xyz+euler xyz、radians、root frame)
        state_prev: (38,) 前 frame の state_current、ep 先頭は None (velocity=zeros)

    Returns:
        state_71: (71,) float32、slice 内訳は module docstring 参照

    Raises:
        ValueError: dim mismatch、NaN/Inf 混入時
    """
    _require_finite("state_current", state_current, SOURCE_STATE_DIM)
    if action_prev is not None:
        _require_finite("action_prev", action_prev, SOURCE_STATE_DIM)
    _require_finite("ee_state", ee_state, SOURCE_EEF_DIM)
    if state_prev is not None:
        _require_finite("state_prev", state_prev, SOURCE_STATE_DIM)

    upper_current = state_current[_UPPER_INDEX]

    joint = upper_current
    if action_prev is None:
        tracking_err = np.zeros(UPPER_BODY_STATE_DIM, dtype=np.float32)
    else:
        tracking_err = action_prev[_UPPER_INDEX] - upper_current
        tracking_err[UPPER19_WAIST_SLICE] = 0.0

    if state_prev is None:
        velocity = np.zeros(UPPER_BODY_STATE_DIM, dtype=np.float32)
    else:
        upper_prev = state_prev[_UPPER_INDEX]
        velocity = upper_current - upper_prev

    hand_state = state_current[SOURCE_ROBOT_Q_DIM : SOURCE_ROBOT_Q_DIM + 2]

    out = np.concatenate(
        [
            joint.astype(np.float32),
            tracking_err.astype(np.float32),
            velocity.astype(np.float32),
            hand_state.astype(np.float32),
            ee_state.astype(np.float32),
        ]
    )
    assert out.shape == (RAMEN_ORI_STATE_DIM,), (
        f"derived state has unexpected shape {out.shape}, expected ({RAMEN_ORI_STATE_DIM},)"
    )
    return out


def derive_state_73d(
    state_current: np.ndarray,
    action_prev: np.ndarray | None,
    ee_state: np.ndarray,
    state_prev: np.ndarray | None,
    depth_contact: np.ndarray,
) -> np.ndarray:
    """1 frame 分の 73D E-β state を組む (E-δ 71D + wrist depth contact bit 2D)。

    Args:
        state_current: (38,)
        action_prev: (38,) or None (区間先頭)
        ee_state: (12,)
        state_prev: (38,) or None (ep 先頭)
        depth_contact: (2,) [left, right] wrist depth contact bit ∈ [0, 1] (soft signal)。
            SGBM+WLS pipeline 未実装時は placeholder zero を渡す。

    Returns:
        state_73: (73,) float32、[0:71] = E-δ、[71:73] = depth_contact
    """
    _require_finite("depth_contact", depth_contact, DEPTH_CONTACT_DIM)
    # E-δ 71D を再利用
    state_71 = derive_state_71d(
        state_current=state_current,
        action_prev=action_prev,
        ee_state=ee_state,
        state_prev=state_prev,
    )
    out = np.concatenate([state_71, depth_contact.astype(np.float32)])
    assert out.shape == (RAMEN_ORI_STATE_DIM_WITH_DEPTH_CONTACT,), (
        f"derived state_73 has unexpected shape {out.shape}"
    )
    return out


def _require_finite(name: str, values: np.ndarray, expected_dim: int) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {values.shape}")
    if values.shape[0] != expected_dim:
        raise ValueError(
            f"expected {expected_dim}-D {name}, got {values.shape[0]}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf")
