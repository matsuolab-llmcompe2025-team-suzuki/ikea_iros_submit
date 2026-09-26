"""Dex1 の把持インターロック (Issue #136、determinded.md D-9)。

力センサが無いので、`hand_state` (実測) と `hand_cmd` (指令) の乖離で
「何かを噛んで止まっている」を判定する。

# 2 つの条件を AND で取る

`IROS2026_RAMEN_suzuki_pick_leg_1` 全 975,291 frame での実測:

| 指令帯 | 追従誤差 median (左 / 右) | 意味 |
|---|---|---|
| 開 (cmd > 3.5) | −0.006 / +0.002 | 追従する = 何も噛んでいない |
| 中間 (2.35–3.5) | −0.006 / +0.013 | 追従する |
| HOLD (1.45–2.35) | **+0.350 / +0.350** | 追従しない = 噛んで止まっている |

**追従誤差だけでは足りない。** 閉じ始めの過渡でも実測は指令に追いつかず
誤差が出るため、`err > 0.2` が最初に立つ瞬間の `hand_state` は median
**4.46** (ほぼ全開) で、HOLD 帯に入っているのは 0.6% しかない。

そこで値の帯も AND する。両方を課すと、右手が掴んだ瞬間の `hand_state` は
median **2.315** (5-95% [2.225, 2.347])、左手は median **2.313** になり、
2114 episode 中 右 2112 / 左 2094 で検出できる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

#: `hand_state` / `hand_cmd` の並び (taskspace_adapter と同じ)。
LEFT: int = 0
RIGHT: int = 1


@dataclass(frozen=True)
class GraspInterlockConfig:
    """把持インターロックの閾値。

    Attributes:
        follow_error_min: `hand_state − hand_cmd` がこれ以上で「追従していない」。
        hold_min / hold_max: 脚を保持しているときの `hand_state` の帯。
    """

    follow_error_min: float = 0.2
    hold_min: float = 1.45
    hold_max: float = 2.35

    def __post_init__(self) -> None:
        if self.follow_error_min <= 0.0:
            raise ValueError(
                f"follow_error_min must be > 0, got {self.follow_error_min}"
            )
        if not self.hold_min < self.hold_max:
            raise ValueError(
                f"hold_min must be < hold_max, got {self.hold_min}, {self.hold_max}"
            )


def is_grasping(
    hand_state: Sequence[float],
    hand_cmd: Sequence[float],
    side: int,
    cfg: Optional[GraspInterlockConfig] = None,
) -> bool:
    """片手が「何かを噛んで止まっている」かを返す。

    Args:
        hand_state: (2,) 実測の Dex1 開度 [left, right]。
        hand_cmd: (2,) 指令の Dex1 開度 [left, right]。
        side: `LEFT` (0) か `RIGHT` (1)。
        cfg: 閾値。None なら default。

    Returns:
        追従していない **かつ** 保持帯に居るなら True。

    Raises:
        ValueError: 長さが 2 でない、side が不正、値が有限でない。
    """
    cfg = cfg or GraspInterlockConfig()
    if side not in (LEFT, RIGHT):
        raise ValueError(f"side must be LEFT(0) or RIGHT(1), got {side}")
    st = _pair(hand_state, "hand_state")
    cmd = _pair(hand_cmd, "hand_cmd")

    value = st[side]
    follow_error = value - cmd[side]
    return (
        follow_error > cfg.follow_error_min
        and cfg.hold_min <= value <= cfg.hold_max
    )


def _pair(value: Sequence[float], label: str) -> tuple[float, float]:
    try:
        items = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be 2 finite floats, got {value!r}") from exc
    if len(items) != 2:
        raise ValueError(f"{label} must have length 2, got {len(items)}")
    for v in items:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"{label} must be finite, got {items}")
    return items[0], items[1]


@dataclass(frozen=True)
class UnmeasuredHandConfig:
    """手の実測が無い経路 (会場の合成 state) の縮退動作 (Codex #2)。

    Attributes:
        phase1_command_settle_sec: Phase 1 で右手の閉指令が把持帯に入ってから
            VLM の確認へ進むまでの時間 [s]。
        phase3_grasp_command_rad: Phase 3 で閉じるときの指令開度 [rad]
            (全閉の探りの代わり)。
        phase3_hand_settle_sec: Phase 3 の手の段で指令後に待つ時間 [s]。
    """

    phase1_command_settle_sec: float
    phase3_grasp_command_rad: float
    phase3_hand_settle_sec: float

    def __post_init__(self) -> None:
        for label in (
            "phase1_command_settle_sec",
            "phase3_grasp_command_rad",
            "phase3_hand_settle_sec",
        ):
            value = float(getattr(self, label))
            if not value > 0.0 or value != value or value == float("inf"):
                raise ValueError(f"unmeasured_hand.{label} must be positive finite")
