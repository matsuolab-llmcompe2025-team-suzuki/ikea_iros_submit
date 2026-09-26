"""腕 kinematics の Protocol 定義 (Issue #123)。

Skill は具象実装ではなくこの Protocol に依存する。これにより
`pixi run test-desktop` (default env) では mock 実装を刺して stage 遷移を
全経路検証でき、実 IK (pinocchio 等の重い依存) を default env に持ち込まずに済む。
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from inference.desktop.lower_policy.kinematics.types import EEPose, IKResult, Side


@runtime_checkable
class ArmKinematics(Protocol):
    """片腕 7-DoF の FK / IK。root_link 固定ベース前提。

    pick_table_leg は下半身がほぼ静止 (骨盤高の変動 median 3mm、root 並進
    median 1.3cm) なので固定ベースで扱える。歩行を伴う skill には使えない。
    """

    def fk(
        self, q: np.ndarray, side: Side, waist: Optional[np.ndarray] = None
    ) -> EEPose:
        """関節角 (7,) [rad] → root_link 基準の EE pose。

        Args:
            waist: 腰 3 関節 (yaw, roll, pitch) [rad]。腕は torso_link に付くので
                root_link 基準の EE を出すにはこれが要る。`None` は全ゼロ扱い。
                dataset 実測では腰の episode 内可動域は median 0.08 rad と小さいが、
                腕の到達長 0.5m では 4cm 相当になり stage の許容 2cm を超えるため、
                実測値を渡すこと。
        """
        ...

    def ik(
        self,
        target: EEPose,
        seed: np.ndarray,
        side: Side,
        waist: Optional[np.ndarray] = None,
    ) -> IKResult:
        """EE pose → 関節角 (7,) [rad]。

        Args:
            target: root_link 基準の目標 pose。
            seed: 反復の初期値 (7,)。通常は現在の関節角を渡す (連続性のため)。
            side: どちらの腕か。
            waist: `fk` と同じ。

        Returns:
            `IKResult`。`status != OK` のとき `q` を使ってはならない。
            解無し / 関節上限 / 特異点を **例外ではなく status で返す** のは、
            stage が abort 判定に使うため (制御ループを例外で壊さない)。
        """
        ...
