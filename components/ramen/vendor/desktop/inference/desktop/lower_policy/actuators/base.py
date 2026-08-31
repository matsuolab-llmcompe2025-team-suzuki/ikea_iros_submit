"""WalkActuator: 移動命令の抽象 interface。

Skill が「どう命令するか」を知らずに前進 / 停止できるように、actuation 経路
(実 SDK / Mock / 将来 Isaac Sim 等) をこの Protocol で差し替え可能にする。

- Skill 側 (skills/) は WalkActuator への依存だけ持つ
- Concrete 実装は actuators/{mock, g1_sdk, ...}.py に置く
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WalkActuator(Protocol):
    """G1 の base locomotion を抽象化した interface。

    実機 G1 の LocoClient と同じセマンティクス:
      - set_velocity: `LocoClient.Move(vx, vy, vyaw)` 相当。1 発呼び出せば
        SDK 内蔵 walk controller が持続 (連続再送は不要)
      - stand_up: `LocoClient.Squat2StandUp()` / `HighStand()` 相当。
        Squat 状態から立ち上がり、以降立ち姿勢を維持
      - damp: `LocoClient.Damp()` 相当。モータをフリーにする。立ち状態で
        呼ぶと崩れ落ちるため、ハーネス支持または安全姿勢を確認した専用手順以外
        からは呼ばない
    """

    def set_velocity(
        self,
        vx: float,
        vy: float,
        vyaw: float,
        *,
        duration: float | None = None,
    ) -> None:
        """base の並進 / yaw 速度を指令する。``duration`` は SDK 側の有効時間 [s]。"""
        ...

    def stand_up(self) -> None:
        """立ち姿勢に遷移する (歩行 balancer は継続)。"""
        ...

    def damp(self) -> None:
        """モータをフリーにする (呼ぶ前に姿勢確保しておくこと)。"""
        ...
