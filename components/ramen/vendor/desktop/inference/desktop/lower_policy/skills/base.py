"""Skill: 個別 skill の共通 lifecycle 定義。

Wrapper が dispatcher 経由で start / stop する単位。skill 内部で thread を
立てるかどうか等の実装は自由 (ABC にした理由: 共通の active flag / 二重
start 保護をここで一括提供したいため)。

step(obs) について:
    per-tick action 生成の hook。Orchestrator の tick loop が Dispatcher
    経由で呼ぶ。default は None を返す (Type A: walk 系のように actuator に
    自前で command を送る skill 用、caller は forward しない)。
    Type B (VLA / GR00T / ACT / diffusion) は step を override して
    action tensor (np.ndarray) を返す (caller が actuator に流し込む)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Skill(ABC):
    """Lower Policy skill の共通 ABC。

    lifecycle:
        instance 化 → (dispatcher が保持) → start → (step per-tick) → stop
        → 再 start OK
        (`start()` は非 idempotent、`stop()` は idempotent、`step()` は任意回)

    subclass の実装義務:
        - class attribute `name: str` を設定 (または __init__ で instance 側)
        - `_on_start(params)` / `_on_stop()` を override
        - per-tick action 生成が要る Type B skill は `step(obs)` も override
    """

    # subclass で必ず override。dispatcher の skill 名 lookup key。
    name: str

    def __init__(self) -> None:
        self._active: bool = False

    @abstractmethod
    def _on_start(self, params: dict) -> None:
        """Skill 固有の起動処理 (`start` から call、外部からは呼ばない)。"""
        ...

    @abstractmethod
    def _on_stop(self) -> None:
        """Skill 固有の停止処理 (`stop` から call、外部からは呼ばない)。"""
        ...

    def start(self, params: dict) -> None:
        """Skill を起動する。既に active なら RuntimeError。"""
        if self._active:
            raise RuntimeError(
                f"skill {self.name!r} already active; call stop() first"
            )
        self._on_start(params)
        self._active = True

    def stop(self) -> None:
        """Skill を停止する (idempotent)。"""
        if not self._active:
            return
        self._on_stop()
        self._active = False

    def step(self, obs: dict) -> Optional[np.ndarray]:
        """per-tick action tensor。default None = 「自前で actuator を叩いた or 何もしない」。

        Type A (walk / stand 系): default のまま。step 内で self._actuator.move(...)
            のように直接 hardware を叩き、caller には None を返す (forward 不要)。
        Type B (VLA / GR00T / ACT / diffusion): override して action tensor を返す。
            caller (Dispatcher / Orchestrator) が actuator に流し込む。
        """
        return None

    @property
    def max_dwell_sec(self) -> Optional[float]:
        """このSkillの最大 dwell 秒 (host 側 fail-safe、Issue #81)。

        Orchestrator が active skill を dispatcher 経由で lookup して読み取り、
        指定秒を超えて active のままなら TRANSITIONS graph の first candidate に
        auto-transition を発火する。

        default `None` = 時間 fail-safe 無し、enter_check ベースの遷移のみ。
        subclass で override して instance level 値を返せる (e.g. MoveToTable は
        constructor で受けた `max_dwell_sec`、SetupSkill は stage 合計時間)。
        """
        return None

    @property
    def is_complete(self) -> bool:
        """Whether a finite, observation-driven skill reached its terminal pose.

        Most skills transition through perception or ``max_dwell_sec`` and
        therefore keep the default ``False``. Collision-aware pre-motion uses
        measured joint convergence instead of a blind timer and overrides it.
        """

        return False

    @property
    def is_active(self) -> bool:
        return self._active
