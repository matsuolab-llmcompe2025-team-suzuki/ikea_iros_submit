"""把持ポーズの供給 interface (Issue #123)。

`pick_table_leg` の stage 1 だけが「毎回変わる目標」を必要とする。dataset 実測では
把持ポーズのばらつきは以下の通りで、**実質 (x, y, yaw) の 3 自由度**しか動かない:

| 自由度 | IQR | 担当 |
|---|---|---|
| z     | 0.026 m   | ほぼ定数 (テーブル面) |
| pitch | 0.166 rad | ほぼ定数 (横倒しの円筒を掴む手首角) |
| x     | 0.047 m   | OBB 中心 |
| y     | 0.077 m   | OBB 中心 |
| yaw   | 0.292 rad | **OBB の角度そのもの** |

したがって YOLO-OBB (`obs["cleaned"]`) + テーブル平面フィットで解ける。
密 depth は不要 (脚がテーブル面上にある拘束で z が決まる)。

本 module は Protocol と、YAML の固定値を返す実装 / test double までを提供する。
OBB からの実装 (`ObbGraspPoseProvider`) はカメラ校正の配線を伴うため別途。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from inference.desktop.lower_policy.kinematics.types import EEPose


@runtime_checkable
class GraspPoseProvider(Protocol):
    """観測から把持ポーズ (root_link 基準) を返す。

    `None` を返した場合、skill は「まだ脚を検出できていない」とみなして
    その tick は動かず待機する (timeout に達したら abort)。
    """

    def grasp_pose(self, obs: dict) -> Optional[EEPose]:
        ...


@dataclass(frozen=True)
class FixedGraspPoseProvider:
    """常に同じ把持ポーズを返す実装。

    用途は 2 つ:
      - CV を繋ぐ前の bring-up (脚を決め打ち位置に置いて stage を通す)
      - unit test

    `available_after_calls` を指定すると、その回数だけ `None` を返してから
    ポーズを返す (検出待ちの経路を試験するため)。
    """

    pose: EEPose
    available_after_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "_calls", 0)

    def grasp_pose(self, obs: dict) -> Optional[EEPose]:
        calls = getattr(self, "_calls", 0)
        object.__setattr__(self, "_calls", calls + 1)
        if calls < self.available_after_calls:
            return None
        return self.pose

    @classmethod
    def from_config(cls, cfg: dict) -> "FixedGraspPoseProvider":
        """`{"pose": [x, y, z, roll, pitch, yaw]}` から構築する。"""
        if not isinstance(cfg, dict) or "pose" not in cfg:
            raise ValueError("fixed grasp provider: 'pose' (6 values) is required")
        return cls(pose=EEPose.from_vec6(cfg["pose"]))


class ScriptedGraspPoseProvider:
    """呼ばれるたびに用意した列を順に返す test double。

    列を使い切ったら最後の値を返し続ける。`None` を混ぜると検出待ちを模擬できる。
    """

    def __init__(self, poses: list[Optional[EEPose]]) -> None:
        if not poses:
            raise ValueError("poses must be non-empty")
        self._poses = list(poses)
        self._i = 0
        self.calls = 0

    def grasp_pose(self, obs: dict) -> Optional[EEPose]:
        self.calls += 1
        pose = self._poses[min(self._i, len(self._poses) - 1)]
        self._i += 1
        return pose


def grasp_pose_from_obb_axis(
    center_xy: np.ndarray,
    long_axis_yaw: float,
    table_z: float,
    *,
    approach_pitch: float,
    roll: float = 0.0,
) -> EEPose:
    """OBB の中心 / 長軸角 / テーブル面高さ から把持ポーズを組む。

    `ObbGraspPoseProvider` を書くときの中核変換をここに切り出しておく
    (カメラ校正に依存しない純幾何なので default env で試験できる)。

    Args:
        center_xy: root_link 基準の把持点 xy [m] (2,)。
        long_axis_yaw: 脚の長軸方向 [rad]。OBB の角度から得る。
        table_z: テーブル面の高さ [m] (root_link 基準)。
        approach_pitch: 横倒しの円筒を掴むための手首 pitch [rad]。
            dataset 実測 median 0.828 (IQR 0.166)。
        roll: 手首 roll [rad]。
    """
    xy = np.asarray(center_xy, dtype=np.float64).reshape(-1)
    if xy.shape != (2,):
        raise ValueError(f"center_xy: must have shape (2,), got {xy.shape}")
    return EEPose(
        position=np.array([xy[0], xy[1], float(table_z)]),
        rpy=np.array([float(roll), float(approach_pitch), float(long_axis_yaw)]),
    )
