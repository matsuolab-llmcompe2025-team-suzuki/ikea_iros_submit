"""MoveToTable: 組立台の前まで歩いて到達する skill。

Wrapper が発火するタイミング:
  - Recipe の step 0 (最初、`inference._deprecated._old_vit_planner.logic_wrapper.recipe.SKILL_SPECS['move_to_table']`)
  - post_condition なし、timing_only モード。判定は wrapper 側の dwell / timeout。
    この Skill 自体は「歩き始め → 止め」の 2 操作のみ。

現状 depth-based safety layer や goal condition (ViT worldstate の
`arm_can_reach`) は非搭載 — 別 Issue で追加予定。
"""

from __future__ import annotations

from typing import Optional

from inference.desktop.lower_policy.actuators.base import WalkActuator
from inference.desktop.lower_policy.skills.base import Skill


# This G1 did not start walking at 0.10 m/s and did at 0.15 m/s. Keep the
# operator-facing defaults in one place so the smoke tool and orchestrator do
# not drift apart.
DEFAULT_WALK_VX = 0.15


class MoveToTable(Skill):
    """G1 が組立台に向かって前進する skill (低速)。

    Skill pattern = Type A (1 発発行完結):
        _on_start で actuator.set_velocity() を 1 回発行し、SDK 側の servo
        controller が持続保持する (`LocoClient.Move(..., continous_move=True)`
        = duration 10 日)。Skill.step は override せず default None (毎 tick で
        何もしない = SDK に同 command を繰り返し送る必要が無い)。次 skill
        遷移で _on_stop が呼ばれ set_velocity(0) で停止。姿勢/FSM遷移は行わない。
        実行前にwalking FSMへ入れるのはoperatorの責務で、公式G1 loco exampleも
        Moveの前後にHighStandを挟まない。詳細は Skill 基底
        class の docstring (base.py) 参照。

    Args:
        actuator: WalkActuator 実装 (Mock / SDK / 等)。
        vx: 前進速度 [m/s]。default 0.15 は実機で歩行確認済みの低速値。
        max_dwell_sec: この Skill の最大 dwell 秒 (Issue #81)。
            - SDK 側 set_velocity の duration にそのまま渡す (firmware fail-safe、
              host 死亡時に速度指令を firmware 側で失効させる)。
            - Orchestrator の `Skill.max_dwell_sec` property 経由で host 側
              auto-transition の閾値にも使う。1 値で host / firmware を同期。
            None なら SDK は継続速度指令 (10 日 duration)、host 側 fail-safe も無し。
    """

    name = "move_to_table"

    def __init__(
        self,
        actuator: WalkActuator,
        vx: float = DEFAULT_WALK_VX,
        max_dwell_sec: Optional[float] = None,
    ) -> None:
        super().__init__()
        self._actuator = actuator
        self._vx = vx
        self._max_dwell_sec = max_dwell_sec

    @property
    def max_dwell_sec(self) -> Optional[float]:
        return self._max_dwell_sec

    def _on_start(self, params: dict) -> None:
        # 前進速度を1発発行。既存walking FSM/stand heightは変更しない。
        # SDK duration も max_dwell_sec と同じ値 (host / firmware fail-safe を同期)。
        self._actuator.set_velocity(
            self._vx, 0.0, 0.0, duration=self._max_dwell_sec
        )

    def _on_stop(self) -> None:
        # 速度ゼロのみ。balancer/FSM/stand heightは既存状態を維持する。
        # 公式StopMoveと同じ1秒timeoutを使い、10日間のzero commandを残さない。
        self._actuator.set_velocity(0.0, 0.0, 0.0, duration=1.0)

    @classmethod
    def from_config(cls, cfg: dict, actuator: WalkActuator) -> "MoveToTable":
        """dict (skill_config.yaml の `skills.move_to_table` セクション) から構築する。

        期待する構造 (Issue #81 Phase 2b):

            {"vx": <float>, "max_dwell_sec": <float or None>}

        `vx` 未指定なら DEFAULT_WALK_VX、`max_dwell_sec` 未指定なら None
        (fail-safe 無し、SDK 側も 10 日 duration)。
        """
        if not isinstance(cfg, dict):
            raise ValueError(
                f"move_to_table config: must be a mapping, got {type(cfg).__name__}"
            )
        vx = float(cfg.get("vx", DEFAULT_WALK_VX))
        raw_dwell = cfg.get("max_dwell_sec")
        max_dwell_sec: Optional[float]
        if raw_dwell is None:
            max_dwell_sec = None
        else:
            max_dwell_sec = float(raw_dwell)
            if max_dwell_sec <= 0:
                raise ValueError(
                    "move_to_table config: max_dwell_sec must be > 0 or null, "
                    f"got {raw_dwell}"
                )
        return cls(actuator, vx=vx, max_dwell_sec=max_dwell_sec)
