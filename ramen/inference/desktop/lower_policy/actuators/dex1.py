"""Dex1-1 グリッパの指令 / 状態 interface と把持判定 (Issue #123)。

## 把持判定の根拠

`Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1` (975,291 frame) で
`hand_state`(実測) と `hand_cmd`(指令) の乖離を調べた結果:

| 指令帯 | 追従誤差 state-cmd (左 / 右) | 意味 |
|---|---|---|
| 開指令 (>3.5)      | -0.006 / +0.002 | 追従する = 何も噛んでいない |
| 中間 (2.35-3.5)    | -0.006 / +0.013 | 追従する |
| HOLD 帯 (1.45-2.35)| **+0.350 / +0.350** | 追従しない = 物を噛んで止まっている |

脚保持中の実測値は 左 2.19 / 右 2.23 (IQR 0.058-0.097)。
つまり **力センサ無しでも「閉じろと言ったのに閉じきらない」ことで把持を検出でき、
さらに止まった位置が脚の径と一致するかで「掴んだのが脚か」まで判別できる**。

この判定を `classify_grasp` に純関数として切り出してある (SDK 非依存、
default env の unit test でしきい値の妥当性を検証できる)。

## 把持時は「全閉」を指令すること (dataset の指令値をそのまま真似ない)

dataset の `hand_cmd` は把持中 1.85 前後に集中しているが、これは
**遠隔操作者のグローブ開度がそのまま記録されているだけ**で、自律制御が真似る
べき値ではない。1.85 を指令すると掴み損ねた時にグリッパが 1.85 に到達してしまい、
`TRANSIT` のまま張り付いて失敗を検出できない。

自律制御では **全閉 (0.0) を指令して脚に機械的に止めさせる**:

    脚あり → 2.2 付近で停止、追従誤差 +2.2  → HOLDING
    空振り → 0.0 まで到達                    → EMPTY_CLOSED
    閉じ途中 → その間                        → TRANSIT

これで把持成否が排他的に判定でき、timeout に頼らずに済む。指令値そのものは
skill 側の YAML (`grip.close_command` / `grip.open_command`) に置く。

## 実機の指令経路について

本 module の `Dex1Gripper` Protocol は指令経路を抽象化する。実装 (`Dex1DdsGripper`)
は cyclonedds / unitree_sdk2py に依存するため **runtime feature-env 専用**、
かつ import は lazy (CLAUDE.md 方針)。Orin/PC2 側で `dex1_1_gripper_server` が
起動している必要がある。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from inference.desktop.lower_policy.kinematics.types import Side


class GraspState(str, Enum):
    """グリッパが今どういう状態か。stage の遷移条件に使う。"""

    OPEN = "open"                    # 開いている (何も持っていない)
    HOLDING = "holding"              # 脚を保持している
    EMPTY_CLOSED = "empty_closed"    # 全閉まで行った = 掴み損ね (空振り)
    TRANSIT = "transit"              # 開閉の途中 / 判定不能


@dataclass(frozen=True)
class GraspThresholds:
    """`classify_grasp` のしきい値。default は dataset 実測由来。

    Attributes:
        open_min: これより大きい実測値は「開いている」。
        hold_state_min / hold_state_max: 脚を保持している時の実測値の帯。
            dataset 実測 左 2.19 / 右 2.23 (IQR 0.058-0.097) を包む幅にしてある。
        fist_max: これより小さい実測値は「空振りの全閉」。
        tracking_err_min: `state - command` がこれを超えたら「何かを噛んでいる」。
            自由運動時の追従誤差は ±0.02 以内、噛んだ時は +0.35 に飽和するので
            その中間を取る。
    """

    open_min: float = 3.5
    hold_state_min: float = 2.05
    hold_state_max: float = 2.35
    fist_max: float = 0.5
    tracking_err_min: float = 0.2

    def __post_init__(self) -> None:
        if not (self.fist_max < self.hold_state_min < self.hold_state_max < self.open_min):
            raise ValueError(
                "GraspThresholds: fist_max < hold_state_min < hold_state_max < "
                f"open_min を満たすこと (got {self})"
            )
        if self.tracking_err_min <= 0:
            raise ValueError("GraspThresholds: tracking_err_min must be > 0")

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "GraspThresholds":
        """YAML section (未指定 key は default) から構築する。"""
        if cfg is None:
            return cls()
        if not isinstance(cfg, dict):
            raise ValueError("grasp_thresholds: must be a mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(
                f"grasp_thresholds: unknown key(s) {sorted(unknown)} "
                f"(valid: {sorted(known)})"
            )
        return cls(**{k: float(v) for k, v in cfg.items()})


def classify_grasp(
    command: float, state: float, th: GraspThresholds = GraspThresholds()
) -> GraspState:
    """指令値と実測値から把持状態を判定する。

    判定順序に意味がある:
      1. 全閉まで到達 → 何も噛んでいない (`EMPTY_CLOSED`)。追従誤差が小さくても
         「閉じきった」事実が優先。
      2. 追従誤差が大きく、かつ止まった位置が脚の径帯 → `HOLDING`。
      3. 開いている → `OPEN`。
      4. それ以外は `TRANSIT` (動作中、まだ判定しない)。

    Args:
        command: 直近に送った指令値 (0=全閉 / 4.5=全開)。
        state: グリッパの実測位置。
    """
    if state < th.fist_max:
        return GraspState.EMPTY_CLOSED
    if (
        (state - command) > th.tracking_err_min
        and th.hold_state_min <= state <= th.hold_state_max
    ):
        return GraspState.HOLDING
    if state > th.open_min:
        return GraspState.OPEN
    return GraspState.TRANSIT


@runtime_checkable
class Dex1Gripper(Protocol):
    """Dex1-1 左右グリッパの指令 / 状態読み出し。

    Skill が自前で保持する (Type A skill が walk actuator を直接叩くのと同じ流儀)。
    こうすることで `Orchestrator._build_obs` を変更せずに済む。
    """

    def command(self, side: Side, position: float) -> None:
        """グリッパへ位置指令を送る (0=全閉 / 4.5=全開)。"""
        ...

    def read(self, side: Side) -> Optional[float]:
        """最新の実測位置。未受信なら `None`。"""
        ...

    def last_command(self, side: Side) -> Optional[float]:
        """直近に送った指令値。未送信なら `None` (`classify_grasp` 用)。"""
        ...

    def close(self) -> None:
        """subscriber / publisher の teardown (idempotent)。"""
        ...


class MockDex1Gripper:
    """`Dex1Gripper` の test double。

    `obstacle_at` を与えると「そこで機械的に止まる」挙動を模擬する
    (= 脚を噛んで閉じきらない)。`None` なら指令に完全追従する (空振り)。

    Args:
        obstacle_at: 左右それぞれの機械停止位置。`None` の side は指令に追従。
        unavailable: `read()` が `None` を返す side (未受信の模擬)。
    """

    def __init__(
        self,
        *,
        obstacle_at: Optional[dict[Side, Optional[float]]] = None,
        unavailable: Optional[set[Side]] = None,
    ) -> None:
        self._obstacle: dict[Side, Optional[float]] = dict(obstacle_at or {})
        self._unavailable: set[Side] = set(unavailable or set())
        self._cmd: dict[Side, float] = {}
        self.closed: bool = False
        # test が指令列を検証できるよう記録する。
        self.command_log: list[tuple[Side, float]] = []

    def set_obstacle(self, side: Side, position: Optional[float]) -> None:
        """途中で「脚を掴んだ / 落とした」を切り替える (stage 試験用)。"""
        self._obstacle[side] = position

    def command(self, side: Side, position: float) -> None:
        self._cmd[side] = float(position)
        self.command_log.append((side, float(position)))

    def read(self, side: Side) -> Optional[float]:
        if side in self._unavailable:
            return None
        cmd = self._cmd.get(side)
        if cmd is None:
            return None
        obstacle = self._obstacle.get(side)
        if obstacle is None:
            return cmd
        # 閉じる方向 (指令 < 障害物位置) では障害物で止まる。開く方向は自由。
        return max(cmd, float(obstacle))

    def last_command(self, side: Side) -> Optional[float]:
        return self._cmd.get(side)

    def close(self) -> None:
        self.closed = True
