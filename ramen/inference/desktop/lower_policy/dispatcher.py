"""SkillDispatchLowerPolicy: skill 名から Skill instance を lookup する
LowerPolicy 実装。

Wrapper (`inference._deprecated._old_vit_planner.logic_wrapper.wrapper.LogicWrapper`) が出力する
`WrapperOutput.active_skill` を Orchestrator 経由で受け取り、対応する
Skill を start / stop する。

interface は `inference._deprecated._old_vit_planner.orchestrator.LowerPolicy` Protocol と一致
(start(skill, params) / stop() の 2 method のみ)。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from inference.desktop.lower_policy.skills.base import Skill

# Wrapper が recipe 完走時に出す sentinel (WrapperOutput.active_skill)。
# dispatcher 側で「stop して以降静止」の意味に写す。
DONE_SENTINEL = "__DONE__"


class SkillDispatchLowerPolicy:
    """skill 名 → Skill instance の routing を行う LowerPolicy 実装。

    Args:
        skills: skill 名 → Skill instance の mapping (registry pattern)。
                dispatcher が扱う全 skill をここで注入する。

    挙動:
        start(skill, params):
            - `skill == DONE_SENTINEL` → 現 active skill を stop、以降 stop 状態
            - 現 active skill と同名 → 何もしない (no-op、wrapper が同 skill を
              CONTINUE 出力してくる想定)
            - 未知 skill 名 → `KeyError` (recipe と registry のズレを早期発見)
            - 通常 → 現 active skill を stop してから新 skill を start
        stop():
            - 現 active skill があれば stop、無ければ何もしない (idempotent)
    """

    def __init__(
        self,
        skills: dict[str, Skill],
        *,
        action_history_maxlen: int = 30,
    ) -> None:
        for key, skill in skills.items():
            if skill.name != key:
                raise ValueError(
                    f"skill.name mismatch: registry key {key!r} vs "
                    f"skill.name {skill.name!r}"
                )
        self._skills: dict[str, Skill] = dict(skills)
        self._active: Optional[Skill] = None
        # 直近 K tick の action を ring buffer で保持 (default 30 = 30Hz で 1s ぶん)。
        # skill 遷移時 snapshot に埋めて resume 時 context 復元に使う。
        self._action_history: deque[np.ndarray] = deque(
            maxlen=action_history_maxlen
        )

    def start(self, skill: str, params: dict) -> None:
        if skill == DONE_SENTINEL:
            self.stop()
            return
        if self._active is not None and self._active.name == skill:
            return
        # 未知 skill 名の検証は「現 active skill を stop する前」に行う。
        # そうしないと、未知 skill を渡された時 caller は KeyError を受ける
        # だけだが実際は「old skill 停止 + new skill 未起動」の中間状態に
        # なり、state 不一致を招く (atomic に振る舞わせるためここで先出し)。
        if skill not in self._skills:
            raise KeyError(f"unknown skill {skill!r}")
        if self._active is not None:
            self._active.stop()
            self._active = None
        self._skills[skill].start(params)
        self._active = self._skills[skill]

    def stop(self) -> None:
        if self._active is None:
            return
        self._active.stop()
        self._active = None

    def step(self, obs: dict) -> Optional[np.ndarray]:
        """active skill の per-tick step を pass-through。

        戻り値の意味は Skill.step と同じ:
            - None: active skill が None、または skill が自前で actuator を叩いた (Type A)
            - np.ndarray: caller が actuator に流す action tensor (Type B の VLA 等)
        """
        if self._active is None:
            return None
        action = self._active.step(obs)
        # 呼び出し側の in-place 変更で history が汚染されないよう copy して保持。
        if action is not None:
            self._action_history.append(action.copy())
        return action

    @property
    def action_history(self) -> list[np.ndarray]:
        """直近 K tick の action snapshot (read-only view、毎回 new list を返す)。

        書き換えは step() 経由のみ、リセットは caller が意図的に行う (現状は無し =
        process 寿命ぶん保持、skill 遷移越しに history が繋がる)。
        """
        return list(self._action_history)

    def restore_action_history(self, actions: list[np.ndarray]) -> None:
        """既存 history をクリアして渡された list で埋め直す (resume 経路用)。

        deque(maxlen) の性質で、渡された actions が maxlen を超える場合は先頭側
        から自動 truncate される。空 list ならクリアだけ。
        """
        self._action_history.clear()
        for a in actions:
            # step() 側と同じく defensive copy (呼び出し側が array を変更しても
            # history が汚染されない)。
            self._action_history.append(a.copy())

    @property
    def active_skill_name(self) -> Optional[str]:
        return self._active.name if self._active is not None else None

    @property
    def active_skill(self) -> Optional[Skill]:
        """現在 active な Skill instance (未 start / 完了時は None)。

        Orchestrator が Skill.max_dwell_sec property を lookup するために公開
        (Issue #81)。skill instance の他 property (state 等) を dispatcher の
        外から読む用途にも使える。書き換えは非対応、start()/stop() 経由のみ。
        """
        return self._active
