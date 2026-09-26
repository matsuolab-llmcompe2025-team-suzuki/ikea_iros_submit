"""腕 kinematics (FK / IK) package (Issue #123)。

`pick_table_leg` は全 stage が「目標ポーズを定めて Motion Planning」で動くため、
EE pose ↔ 関節角の変換が必須。下半身がほぼ静止する skill 専用の
**固定ベース片腕 7-DoF** kinematics を提供する。

- `types`: EEPose / IKResult / IKStatus / Side (SDK 非依存、default env で import 可)
- `base`: `ArmKinematics` Protocol
- `mock`: unit test 用の可逆 test double
- `dls`: damped least squares による実装 (Issue #123)

具象実装ではなく `ArmKinematics` Protocol に依存すること。
"""

from inference.desktop.lower_policy.kinematics.base import ArmKinematics
from inference.desktop.lower_policy.kinematics.types import (
    NUM_ARM_JOINTS_PER_SIDE,
    EEPose,
    IKResult,
    IKStatus,
    Side,
    matrix_to_rpy,
    rpy_to_matrix,
)

__all__ = [
    "NUM_ARM_JOINTS_PER_SIDE",
    "ArmKinematics",
    "EEPose",
    "IKResult",
    "IKStatus",
    "Side",
    "matrix_to_rpy",
    "rpy_to_matrix",
]
