"""Waist actuator interface + Mock 実装 + G1 real 実装 (Issue #125 Phase 9a / B-1)。

G1 body joints layout:
    [0-11]  = legs (2 x 6 joints)
    [12-14] = waist (yaw, roll, pitch)
    [15-28] = arms (2 x 7 joints)

RAMEN-Ori / GR00T 両方の action 19D = waist(3) + arm(14) + hand(2)。本 module
は waist 3 joints (indices 12-14) を SDK に送る layer を担当。

# 実装方針

- 実 G1 waist 制御は `G1ArmActuator` と同じ rt/arm_sdk topic に載せる (arm と
  waist が同 LOW_CMD、1 publisher/1 topic で DDS 衝突無し)。Phase B-1 (Issue
  #125) で `G1ArmActuator._build_lowcmd` を waist 3-slot override に拡張済み。
- 本 module の `G1WaistActuator` は共有 `G1ArmActuator` instance への thin
  wrapper: `send_action(waist_3d)` が内部で `arm_actuator.send_waist_action(...)`
  を呼ぶだけ。arm と waist は publisher (250Hz thread) を共有。

# Protocol

`send_action(positions: Sequence[float])` の 1 method のみ (3D 想定)。future
real 実装で SDK 側の rate / control-mode handshake を追加する場合、Protocol
は methodname を維持したまま実装内部で拡張する形。
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence


G1_NUM_WAIST_JOINTS: int = 3
G1_WAIST_JOINT_INDICES: tuple[int, ...] = (12, 13, 14)  # yaw / roll / pitch


class WaistActuator(Protocol):
    """Waist 3-joint action sink (RAMEN-Ori / GR00T の action[0:3] 対応)。"""

    def send_action(self, positions: Sequence[float]) -> None:
        """Send waist 3-joint target positions [rad]。

        Args:
            positions: length-3 sequence、G1_WAIST_JOINT_INDICES 順
                (yaw, roll, pitch)。
        """
        ...


class MockWaistActuator:
    """DDS 送信せずに直近 target を保持する mock。tests + entrypoint dry-run 用。

    Attributes:
        history: send_action() 呼出履歴 (append-only list of tuple)。
        latest: 直近の target positions or None。
    """

    def __init__(self) -> None:
        self.history: list[tuple[float, ...]] = []
        self.latest: tuple[float, ...] | None = None

    def send_action(self, positions: Sequence[float]) -> None:
        arr = tuple(float(p) for p in positions)
        if len(arr) != G1_NUM_WAIST_JOINTS:
            raise ValueError(
                f"waist positions must have length {G1_NUM_WAIST_JOINTS}, got {len(arr)}"
            )
        self.history.append(arr)
        self.latest = arr


class G1WaistActuator:
    """G1 real waist actuator (Phase B-1)。共有 G1ArmActuator に waist target を dispatch。

    ## Architecture

    G1 の rt/arm_sdk topic は arm 14 joint と waist 3 joint を同一 LowCmd に載せる
    ため、arm と waist で publisher 分離すると DDS で衝突する。よって本 class は
    **既存の G1ArmActuator instance を共有** し、`send_waist_action` を経由して
    同じ publish thread から waist も送出する。arm 側 send_action と waist 側
    send_action は独立 (arm と waist を別 timing で更新可)、latest snapshot が
    同一 lowcmd に載る形。

    ## 使い方

    ```python
    arm_actuator = G1ArmActuator(...)
    arm_actuator.start()
    waist_actuator = G1WaistActuator(arm_actuator)  # 共有 = 別 publisher なし
    # VlaSkill 側:
    #   arm 14D → arm_actuator.send_action(arm_14d)
    #   waist 3D → waist_actuator.send_action(waist_3d) → arm_actuator.send_waist_action
    ```

    ## 単発 verify

    `G1ArmActuator.read_waist_positions()` で実機 lowstate から waist 3-D pose を
    取れる (diagnostic 用)。SDK 側で waist joints が指定 rad に追従することを目視で
    確認するのが hardware verify (ハーネス済想定)。

    Attributes:
        _arm_actuator: 共有する G1ArmActuator instance。send_waist_action を呼ぶ。
    """

    def __init__(self, arm_actuator: Any) -> None:
        """
        Args:
            arm_actuator: 既に init 済 & start 済の G1ArmActuator (or 互換 duck-type)。
                          `send_waist_action(positions)` method を持つ必要あり。
        """
        if not hasattr(arm_actuator, "send_waist_action"):
            raise TypeError(
                "arm_actuator must have send_waist_action method "
                "(expected G1ArmActuator with Phase B-1 waist extension)"
            )
        self._arm_actuator = arm_actuator

    def send_action(self, positions: Sequence[float]) -> None:
        """Send waist 3-joint target [rad] to shared G1ArmActuator publisher。"""
        arr = tuple(float(p) for p in positions)
        if len(arr) != G1_NUM_WAIST_JOINTS:
            raise ValueError(
                f"waist positions must have length {G1_NUM_WAIST_JOINTS}, got {len(arr)}"
            )
        self._arm_actuator.send_waist_action(arr)
