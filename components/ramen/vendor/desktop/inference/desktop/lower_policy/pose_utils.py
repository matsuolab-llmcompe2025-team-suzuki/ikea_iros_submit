"""14-D 腕関節 pose の共通定数 / helper (Issue #81 Phase 2b)。

複数 skill (SetupSkill / SampleVLASkill / ...) が同じ sparse pose 記法
(joint short label → value dict) を YAML から読み込むための shared 定義。
skill 間で joint 名や densify logic を duplicate しないよう共通化する。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# 14-D arm joint の short label。順序 = G1_ARM_JOINT_INDICES (SDK motor 15..28) と一致。
# YAML の pose_rad 記法で使う joint 名の master list。
JOINT_NAMES: tuple[str, ...] = (
    "L.shoulder_pitch", "L.shoulder_roll", "L.shoulder_yaw",
    "L.elbow", "L.wrist_roll", "L.wrist_pitch", "L.wrist_yaw",
    "R.shoulder_pitch", "R.shoulder_roll", "R.shoulder_yaw",
    "R.elbow", "R.wrist_roll", "R.wrist_pitch", "R.wrist_yaw",
)
_JOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}

NUM_ARM_JOINTS: int = len(JOINT_NAMES)

# G1ArmActuator の ArmSafetyLimits.joint_max_abs と揃える (異常値ガード)。
POSE_ABS_LIMIT_RAD: float = 1.5


def densify_pose(
    sparse: Optional[dict[str, Any]], context: str = ""
) -> np.ndarray:
    """sparse dict ({joint_name: value}) を (14,) dense array に展開する。

    明記されない joint は 0.0 rad。unknown joint 名は error。
    None / 空 dict はゼロ姿勢 (全 joint 0.0) として扱う。

    Args:
        sparse: `{"L.shoulder_roll": 1.2, ...}` の dict、または None。value は
            `float(value)` で数値化するので int / str-castable でも通る。
        context: error message 用の呼び出し元 hint (skill / stage 名等)。
    """
    if sparse is None:
        sparse = {}
    if not isinstance(sparse, dict):
        raise ValueError(
            f"pose_rad{f' ({context})' if context else ''}: "
            f"must be a mapping, got {type(sparse).__name__}"
        )
    dense = np.zeros(NUM_ARM_JOINTS, dtype=np.float64)
    for name, value in sparse.items():
        if name not in _JOINT_INDEX:
            raise ValueError(
                f"pose_rad{f' ({context})' if context else ''}: "
                f"unknown joint {name!r} (valid: {list(JOINT_NAMES)})"
            )
        dense[_JOINT_INDEX[name]] = float(value)
    return dense


def validate_pose_bounds(pose: np.ndarray, context: str = "") -> None:
    """pose が NaN/Inf を含まず、±POSE_ABS_LIMIT_RAD 以内であることを assert する。

    G1ArmActuator の safety clamp と同じ ±1.5 rad を超えると `ValueError`。
    """
    if pose.shape != (NUM_ARM_JOINTS,):
        raise ValueError(
            f"pose{f' ({context})' if context else ''}: "
            f"must have shape ({NUM_ARM_JOINTS},), got {pose.shape}"
        )
    if not np.all(np.isfinite(pose)):
        raise ValueError(
            f"pose{f' ({context})' if context else ''}: contains NaN/Inf"
        )
    if float(np.abs(pose).max()) > POSE_ABS_LIMIT_RAD:
        raise ValueError(
            f"pose{f' ({context})' if context else ''}: "
            f"exceeds safety limit ±{POSE_ABS_LIMIT_RAD} rad"
        )
