"""pick_table_leg ハイブリッドの本体 (Issue #136)。

boundary の `Policy.act()` から呼ばれることを想定した、区間管理 + 実行系の
振り分け。**提出テンプレート (`ikea_iros_submit`) はまだ repo に無い**ので
(#116 の vendor 未着手)、ここは transport 非依存の純粋なロジックとして書く。
テンプレートが入ったら `Policy.act()` から `step()` を呼ぶだけで繋がる。

# 1 tick の流れ

    step(obs)
      ├ 区間 1 (VLA)  : VLA に投げる + 1→2 の境界を検出
      ├ 区間 2 (MP)   : 補間した (25,) を返す + 実測到達で 2→3
      └ 区間 3 (VLA)  : VLA に投げるだけ (境界検出なし = 最終区間)

VLA そのものは呼ばない。`vla_step` として注入する
(GR00T pick_legs は Python 3.12 の別プロセス worker 経由で、生成コストが
大きいため区間ごとに起動しない — determinded.md D-1 制約 7)。

# 単調性

フェーズを進めるのは `PhaseStateMachine.advance()` だけ。境界判定が何を
返しても「進む」以外は起きない (D-9)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from inference.desktop.pick_leg_hybrid.boundary import (
    BoundaryDecision,
    GraspBoundaryDetector,
)
from inference.desktop.pick_leg_hybrid.config import PickLegHybridConfig
from inference.desktop.pick_leg_hybrid.phase2 import EePose, Phase2Motion
from inference.desktop.pick_leg_hybrid.phases import (
    DESCRIPTION,
    Phase,
    PhaseStateMachine,
)

#: VLA を呼ぶ関数の型。obs dict → boundary `(T, 25)` chunk。
VlaStep = Callable[[dict], np.ndarray]

#: 実測の手先姿勢を得る関数の型。obs dict → (左, 右)。
EeProbe = Callable[[dict], "tuple[EePose, EePose]"]


@dataclass
class TickResult:
    """1 tick の結果 (そのまま JSONL に流せる)。"""

    phase: Phase
    executor: str
    action: Optional[np.ndarray] = None
    advanced: bool = False
    boundary: Optional[BoundaryDecision] = None
    reached: bool = False
    note: str = ""


class PickLegHybridController:
    """3 区間を管理し、区間ごとに実行系へ振り分ける。

    Args:
        cfg: 設定 (`config.load_yaml` で読む)。
        vla_step: VLA を呼ぶ関数。obs → `(T, 25)`。
        ee_probe: 実測の手先姿勢を返す関数。obs → (左, 右)。
            区間 2 の始点取得と到達判定に使う。
        detector: 1→2 の境界検出器。None なら cfg から作る。
    """

    def __init__(
        self,
        cfg: PickLegHybridConfig,
        *,
        vla_step: VlaStep,
        ee_probe: EeProbe,
        detector: Optional[GraspBoundaryDetector] = None,
    ) -> None:
        self.cfg = cfg
        self._vla_step = vla_step
        self._ee_probe = ee_probe
        self.state = PhaseStateMachine()
        self.motion = Phase2Motion(cfg.phase2)
        self.detector = detector or GraspBoundaryDetector(
            cfg=cfg.boundary,
            interlock_cfg=cfg.interlock,
            vlm_cfg=cfg.vlm,
        )

    def reset(self) -> None:
        """エピソード境界。boundary の `Policy.reset()` から呼ぶ。"""
        self.state.reset()
        self.motion.reset()
        self.detector.reset()

    def step(self, obs: dict) -> TickResult:
        """1 tick 進める。

        Args:
            obs: 観測 dict。以下を読む:
                - ``t``            : 時刻 [s]
                - ``hand_state``   : (2,) Dex1 実測  (区間 1 のみ)
                - ``hand_cmd``     : (2,) Dex1 指令  (区間 1 のみ)
                - ``images_b64``   : overlay 済み [過去…, 現在] (区間 1 のみ)
                その他の key は `vla_step` / `ee_probe` に素通しする。

        Returns:
            `TickResult`。`action` は boundary `(T, 25)`、または VLA の戻り値。
        """
        phase = self.state.phase
        if phase is Phase.APPROACH_GRASP:
            return self._step_phase1(obs)
        if phase is Phase.CARRY_TO_LEFT:
            return self._step_phase2(obs)
        return self._step_phase3(obs)

    # ------------------------------------------------------------ 区間 1 (VLA)

    def _step_phase1(self, obs: dict) -> TickResult:
        action = self._vla_step(obs)
        decision = self.detector.update(
            t=float(obs["t"]),
            hand_state=obs["hand_state"],
            hand_cmd=obs["hand_cmd"],
            images_b64=obs.get("images_b64", ()),
            current_phase_text=DESCRIPTION[Phase.APPROACH_GRASP],
        )
        out = TickResult(
            phase=Phase.APPROACH_GRASP,
            executor="vla",
            action=action,
            boundary=decision,
        )
        if decision.fire:
            # 遷移した瞬間の実姿勢を区間 2 の始点として捉える (可変始点、D-10)
            left, right = self._ee_probe(obs)
            self.motion.start(float(obs["t"]), left, right)
            out.advanced = self.state.advance()
            out.note = "1→2: 把持インターロック AND VLM が成立"
        return out

    # ------------------------------------------------------------ 区間 2 (MP)

    def _step_phase2(self, obs: dict) -> TickResult:
        t = float(obs["t"])
        if not self.motion.started:
            # 区間 1 を経ずに 2 から始まった場合 (再開など) の保険
            left, right = self._ee_probe(obs)
            self.motion.start(t, left, right)
        action = self.motion.step(t)
        out = TickResult(
            phase=Phase.CARRY_TO_LEFT,
            executor="mp",
            action=action[None, :],   # (1, 25) chunk として返す
        )
        # 到達判定は**実測**で行う (指令どおり動いていない場合に誤って進まないため)
        left, right = self._ee_probe(obs)
        out.reached = self.motion.reached(left, right)
        if out.reached:
            out.advanced = self.state.advance()
            out.note = "2→3: 実測 EE pose が終点の許容内に入った"
        return out

    # ------------------------------------------------------------ 区間 3 (VLA)

    def _step_phase3(self, obs: dict) -> TickResult:
        return TickResult(
            phase=Phase.HANDOVER_ONWARD,
            executor="vla",
            action=self._vla_step(obs),
            note="最終区間 (境界検出なし)",
        )
