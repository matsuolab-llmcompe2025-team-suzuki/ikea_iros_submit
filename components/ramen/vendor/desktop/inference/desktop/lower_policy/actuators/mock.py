"""MockWalkActuator: SDK 無しで Skill をテストするための stub 実装。

呼び出された actuator method と引数を events list に貯めるだけ。
Skill の unit test / orchestrator smoke で使う。
"""

from __future__ import annotations


class MockWalkActuator:
    """呼び出しを list に記録するだけの WalkActuator 実装 (副作用ゼロ)。

    Attributes:
        events: 呼び出し履歴。tuple の先頭要素が method 名、続く要素が引数。
                例: [("set_velocity", 0.1, 0.0, 0.0), ("stand_up",)]
    """

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def set_velocity(
        self,
        vx: float,
        vy: float,
        vyaw: float,
        *,
        duration: float | None = None,
    ) -> None:
        event = ("set_velocity", vx, vy, vyaw)
        if duration is not None:
            event += (duration,)
        self.events.append(event)

    def stand_up(self) -> None:
        self.events.append(("stand_up",))

    def damp(self) -> None:
        self.events.append(("damp",))
