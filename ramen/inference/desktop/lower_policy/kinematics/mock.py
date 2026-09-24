"""MockArmKinematics: unit test 用の解析的に可逆な擬似 kinematics (Issue #123)。

実機の G1 腕とは無関係な「位置 = 関節角の一部」という自明な写像を使う。
狙いは stage 遷移ロジックの検証であって kinematics の正しさではないため、
`fk(ik(pose)) == pose` が厳密に成り立つことだけを保証する。

    fk : position = q[0:3] + side_offset,  rpy = q[3:6]   (q[6] は冗長軸、未使用)
    ik : 上記の逆

`force_status` / `unreachable_radius` で abort 経路 (IK 失敗) も試験できる。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from inference.desktop.lower_policy.kinematics.types import (
    NUM_ARM_JOINTS_PER_SIDE,
    EEPose,
    IKResult,
    IKStatus,
    Side,
)

# 左右の EE を分離するためのオフセット [m]。dataset の左右 y 符号
# (左 y>0 / 右 y<0) と揃えておくと test の可読性が上がる。
_SIDE_OFFSET: dict[Side, np.ndarray] = {
    Side.LEFT: np.array([0.0, 0.15, 0.0]),
    Side.RIGHT: np.array([0.0, -0.15, 0.0]),
}


class MockArmKinematics:
    """`ArmKinematics` Protocol の test double。

    Args:
        joint_limit_rad: |q| がこれを超えたら `JOINT_LIMIT` を返す。
        unreachable_radius: root からの距離がこれを超える target は `NOT_CONVERGED`。
            None なら距離チェックをしない。
        force_status: 指定すると ik() が常にこの status を返す (abort 経路の試験用)。
    """

    def __init__(
        self,
        *,
        joint_limit_rad: float = 1.5,
        unreachable_radius: Optional[float] = None,
        force_status: Optional[IKStatus] = None,
    ) -> None:
        self.joint_limit_rad = float(joint_limit_rad)
        self.unreachable_radius = unreachable_radius
        self.force_status = force_status
        # test が「何回 ik を呼んだか」を検証できるよう記録する。
        self.ik_calls: list[tuple[EEPose, Side]] = []

    def fk(
        self, q: np.ndarray, side: Side, waist: Optional[np.ndarray] = None
    ) -> EEPose:
        # mock は腰を無視する (署名互換のためだけに受ける)
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape != (NUM_ARM_JOINTS_PER_SIDE,):
            raise ValueError(
                f"fk: q must have shape ({NUM_ARM_JOINTS_PER_SIDE},), got {arr.shape}"
            )
        return EEPose(position=arr[0:3] + _SIDE_OFFSET[side], rpy=arr[3:6])

    def ik(
        self,
        target: EEPose,
        seed: np.ndarray,
        side: Side,
        waist: Optional[np.ndarray] = None,
    ) -> IKResult:
        self.ik_calls.append((target, side))
        seed_arr = np.asarray(seed, dtype=np.float64).reshape(-1)
        if seed_arr.shape != (NUM_ARM_JOINTS_PER_SIDE,):
            raise ValueError(
                f"ik: seed must have shape ({NUM_ARM_JOINTS_PER_SIDE},), "
                f"got {seed_arr.shape}"
            )

        q = np.zeros(NUM_ARM_JOINTS_PER_SIDE, dtype=np.float64)
        q[0:3] = target.position - _SIDE_OFFSET[side]
        q[3:6] = target.rpy
        q[6] = seed_arr[6]  # 冗長軸は seed を引き継ぐ (連続性の模擬)

        if self.force_status is not None and self.force_status is not IKStatus.OK:
            return IKResult(
                status=self.force_status,
                q=q,
                position_error=float("inf"),
                rotation_error=float("inf"),
                iterations=0,
            )

        if (
            self.unreachable_radius is not None
            and float(np.linalg.norm(target.position)) > self.unreachable_radius
        ):
            return IKResult(
                status=IKStatus.NOT_CONVERGED,
                q=q,
                position_error=float("inf"),
                rotation_error=float("inf"),
                iterations=0,
            )

        if float(np.abs(q).max()) > self.joint_limit_rad:
            return IKResult(
                status=IKStatus.JOINT_LIMIT,
                q=q,
                position_error=0.0,
                rotation_error=0.0,
                iterations=1,
            )

        return IKResult(
            status=IKStatus.OK,
            q=q,
            position_error=0.0,
            rotation_error=0.0,
            iterations=1,
        )
