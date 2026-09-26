"""SafeActuator: manipulation actuator を safety filter で wrap する。

将来の skill (GR00T / VLA / manipulation policy) が arm/hand joint 出力を
SDK に送る際、joint 過大値 / 下半身 誤動作 / hand 過大値 を防ぐ runtime
monitor 層。CLAUDE.md の 4-layer safety のうち runtime monitor に相当。

## 使い方 (将来 manipulation actuator 実装後)

    raw = G1SDKManipulationActuator(...)   # 未実装、別 Issue
    safe = SafeActuator(raw)               # default limits で wrap
    safe.send_joint_trajectory(names, positions)  # 自動で clamp + block

## 現時点

manipulation actuator が存在しないので実 use 先無し。tests は MockInner
に対する unit test のみ。skill 側実装時に SafeActuator を経由するよう
組む契約とする。

## 由来

もともと `_deprecated/g1_motion_adapter/g1_motion_adapter/motion_adapter_node.py`
(ROS2 node) が持っていた safety filter を、topology β (Desktop → SDK direct)
に合わせて ROS2 依存を剥がした純 Python として再実装したもの。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


# G1 の下半身 12 joint (2 脚 x 6 joint)。upper body assembly 中は動かさない。
DEFAULT_BLOCKED_JOINTS: frozenset[str] = frozenset(
    f"g1_joint_{index:02d}" for index in range(12)
)


@dataclass(frozen=True)
class SafetyLimits:
    """SafeActuator の閾値設定。

    Attributes:
        joint_max_abs: joint position の絶対値上限 [rad]。default 1.5 は
            motion_adapter (`_deprecated/g1_motion_adapter/...`) の default 継承。
        hand_max_abs: hand grip position の絶対値上限。default 1.0 は同上。
        blocked_joints: publish から除外する joint 名の集合。default は
            `DEFAULT_BLOCKED_JOINTS` (G1 下半身 12 joint)。
    """

    joint_max_abs: float = 1.5
    hand_max_abs: float = 1.0
    blocked_joints: frozenset[str] = field(default_factory=lambda: DEFAULT_BLOCKED_JOINTS)


def _clamp(value: float, limit: float) -> float:
    lim = abs(float(limit))
    return max(-lim, min(lim, float(value)))


class SafeActuator:
    """任意の manipulation actuator を safety filter で wrap する。

    inner は duck typing で以下 method を持つことを期待:
      - `send_joint_trajectory(names: Sequence[str], positions: Sequence[float])`
      - `send_hand(positions: Sequence[float])`

    Protocol を定義しないのは、対応する manipulation actuator が未実装で
    method signature が固まらないため (premature abstraction 回避)。実
    actuator 実装時に Protocol 化を検討する。

    Args:
        inner: safety filter を経由して呼び出す先の actuator。
        limits: 閾値設定。None なら `SafetyLimits()` (default 値) を使う。
    """

    def __init__(self, inner: object, limits: SafetyLimits | None = None) -> None:
        self._inner = inner
        self._limits = limits if limits is not None else SafetyLimits()

    def send_joint_trajectory(
        self, names: Sequence[str], positions: Sequence[float]
    ) -> None:
        """joint 命令を送信 (blocked joint 除外 + abs clamp 済み)。

        全 joint が blocked に該当した場合、inner を呼ばず drop する
        (motion_adapter と同じ挙動。空 trajectory を送るのは意味が無い)。
        """
        if len(names) != len(positions):
            raise ValueError(
                f"names ({len(names)}) と positions ({len(positions)}) の長さ不一致"
            )
        keep_idx = [i for i, n in enumerate(names) if n not in self._limits.blocked_joints]
        if not keep_idx:
            return
        safe_names = [names[i] for i in keep_idx]
        safe_positions = [_clamp(positions[i], self._limits.joint_max_abs) for i in keep_idx]
        self._inner.send_joint_trajectory(safe_names, safe_positions)  # type: ignore[attr-defined]

    def send_hand(self, positions: Iterable[float]) -> None:
        """hand grip 命令を送信 (abs clamp 済み)。"""
        clamped = [_clamp(v, self._limits.hand_max_abs) for v in positions]
        self._inner.send_hand(clamped)  # type: ignore[attr-defined]
