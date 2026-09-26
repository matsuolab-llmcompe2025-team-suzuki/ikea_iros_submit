"""pick_table_leg の 3 区間と、後戻りしない状態機械 (Issue #136)。

区間 (determinded.md D-4):

    1 APPROACH_GRASP   脚に近づいて右手で掴む            VLA
    2 CARRY_TO_LEFT    右手で脚を左手の位置まで運ぶ        Motion planning
    3 HANDOVER_ONWARD  左右で掴み方を調整し、そのまま最後まで  VLA

VLA の流れの中に Motion planning が 1 区間だけ挟まる形。境界は 2 箇所。

# 単調性はここで保証する (D-9)

`advance()` 以外でフェーズは変わらない。「戻る」API を持たない。
VLM に現在フェーズを渡すのは精度のための文脈であって、単調性をモデルの
挙動に依存させない。境界判定が何を返しても、状態機械が受け付けるのは
「次へ進む」だけ。
"""

from __future__ import annotations

from enum import IntEnum


class Phase(IntEnum):
    """pick_table_leg の 3 区間。値は 1 始まり (人間の説明と一致させる)。"""

    APPROACH_GRASP = 1
    CARRY_TO_LEFT = 2
    HANDOVER_ONWARD = 3


#: 各区間をどの実行系が回すか。
EXECUTOR: dict[Phase, str] = {
    Phase.APPROACH_GRASP: "vla",
    Phase.CARRY_TO_LEFT: "mp",
    Phase.HANDOVER_ONWARD: "vla",
}

#: 人間向けの説明 (log / prompt 組み立てに使う)。
DESCRIPTION: dict[Phase, str] = {
    Phase.APPROACH_GRASP: "脚に近づいて右手で掴む",
    Phase.CARRY_TO_LEFT: "右手で脚を左手の位置まで運ぶ",
    Phase.HANDOVER_ONWARD: "左右の手で掴み方を調整し、そのまま最後まで",
}

LAST_PHASE: Phase = Phase.HANDOVER_ONWARD


class PhaseStateMachine:
    """後戻りしないフェーズ管理。

    `advance()` だけがフェーズを進める。最終区間で `advance()` を呼んでも
    進まず False を返す (例外にはしない — 境界判定が最終区間で発火し続けても
    無害に握り潰したいため)。
    """

    def __init__(self, phase: Phase = Phase.APPROACH_GRASP) -> None:
        self._phase = Phase(phase)
        self._advances = 0

    @property
    def phase(self) -> Phase:
        """現在の区間。"""
        return self._phase

    @property
    def executor(self) -> str:
        """現在の区間を回す実行系 (`"vla"` / `"mp"`)。"""
        return EXECUTOR[self._phase]

    @property
    def advance_count(self) -> int:
        """これまでに進んだ回数 (log 用)。"""
        return self._advances

    @property
    def is_last(self) -> bool:
        """最終区間にいるか。"""
        return self._phase is LAST_PHASE

    def advance(self) -> bool:
        """次の区間へ進む。進めたら True、最終区間なら False。"""
        if self.is_last:
            return False
        self._phase = Phase(int(self._phase) + 1)
        self._advances += 1
        return True

    def reset(self) -> None:
        """区間 1 に戻す。**エピソード境界専用**。

        boundary の `Policy.reset()` から呼ぶ。実行中の後戻りには使わない
        (実行中に戻す API は意図的に持たせていない)。
        """
        self._phase = Phase.APPROACH_GRASP
        self._advances = 0

    def __repr__(self) -> str:
        return (
            f"PhaseStateMachine(phase={self._phase.name}, "
            f"executor={self.executor!r}, advances={self._advances})"
        )
