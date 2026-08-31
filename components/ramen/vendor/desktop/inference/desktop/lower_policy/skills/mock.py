"""MockSkill: e2e smoke 用の dummy Skill 実装。

Issue #47 の Orchestrator core (P6e) + smoke (P6f) では 6 skill (move_to_table /
move_table_base / pick_table_leg / insert_table_leg / rotate_leg_to_tighten /
flip_table) の全 registry を MockSkill で埋め、hardware / VLA 依存無しで純粋な
fire event → dispatcher.start() の flow を検証する。

本実装 (MoveToTable の step 型化 / Gr00tSkill / ActSkill 等) が 1 個ずつ登場する
たびに registry の該当 entry を差し替える。全 6 skill が本実装になったら
`mock.py` は削除予定 (Issue #47 の scope 外、別 Epic で完了)。
"""

from __future__ import annotations

from typing import Optional

from inference.desktop.lower_policy.skills.base import Skill


class MockSkill(Skill):
    """no-op dummy Skill。start/stop は空、step は default None。

    dispatcher の registry key と `name` が一致する必要があるので、instance
    ごとに name を指定できるよう instance attribute で受ける。

    Usage:
        registry = {name: MockSkill(name) for name in [
            "move_to_table", "move_table_base", "pick_table_leg", ...
        ]}

    Args:
        skill_name: dispatcher registry key と一致させる名前。
        max_dwell_sec: この Mock instance が主張する dwell 上限。orchestrator の
            auto-transition test で使う (Issue #81)。default `None` (fail-safe 無し)。
    """

    def __init__(
        self, skill_name: str, max_dwell_sec: Optional[float] = None
    ) -> None:
        super().__init__()
        self.name = skill_name
        self._max_dwell_sec = max_dwell_sec

    @property
    def max_dwell_sec(self) -> Optional[float]:
        return self._max_dwell_sec

    def _on_start(self, params: dict) -> None:
        # 意図的に no-op。log は Orchestrator の JSONL sink 責務。
        pass

    def _on_stop(self) -> None:
        pass
