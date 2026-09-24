"""SetupSkill: 実 G1 で orchestrator 起動直後に腕を安全 clearance 姿勢へ持って
いくための Type B skill (Issue #81)。

## 目的

Issue #81 で entrypoint の `_run_pre_motion` (Skill 階層外の private helper) を
Skill 化。orchestrator の initial skill として dispatch されると、YAML で
定義された複数 stage を順に流し、指定 dwell 経過後に最終 stage を hold し続ける。
`max_dwell_sec` property (= `total_dwell_sec`) を expose することで、
Orchestrator が `dispatcher.active_skill.max_dwell_sec` 経由で読み取り、
全 stage 消化後に move_to_table への auto-transition を発火する (Phase 3)。

## Type B contract

- `step(obs) -> np.ndarray (14,)`: 現在時刻の elapsed sec に応じて current stage
  の pose を返す。最終 stage 消化後は最終 stage pose を返し続ける
  (arm_actuator はその pose を hold し続ける)。
- `obs` は使わない (時間ベースの deterministic 動作)。VLA と違って観測依存無し。

## YAML

`from_yaml(path)` で読み込む YAML shape は
`inference/desktop/lower_policy/configs/skill_config.yaml` 参照:

    skills:
      setup:
        stages:
          - name: <str>
            dwell_sec: <positive float>
            pose_rad: {<joint_name>: <float>, ...}   # sparse (指定外は 0.0)

`joint_name` は G1_ARM_JOINT_INDICES 15..28 に対応する 14 個の short label
(例: `L.shoulder_pitch`, `R.elbow`)。詳細は `JOINT_NAMES` 参照。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml

from inference.desktop.lower_policy.pose_utils import (
    JOINT_NAMES,
    NUM_ARM_JOINTS,
    densify_pose,
    validate_pose_bounds,
)
from inference.desktop.lower_policy.skills.base import Skill


@dataclass(frozen=True)
class SetupStage:
    """SetupSkill の 1 stage 定義。pose_rad は (14,) の numpy array (dense)。"""

    name: str
    dwell_sec: float
    pose_rad: np.ndarray  # shape=(14,)

    def __post_init__(self) -> None:
        if self.dwell_sec <= 0:
            raise ValueError(
                f"stage {self.name!r}: dwell_sec must be > 0, got {self.dwell_sec}"
            )
        # pose_utils の共通 validator で shape / NaN / ±1.5 rad を一括 check。
        validate_pose_bounds(self.pose_rad, context=f"stage {self.name!r}")


class SetupSkill(Skill):
    """時間ベース多段 pose Skill (Type B、Issue #81)。

    stages を順に流し、各 stage の dwell_sec 経過後に次 stage へ切り替える。
    最終 stage 経過後は最終 pose を返し続ける (arm_actuator が hold)。

    Args:
        stages: 実行順 stage list。空 list は不可。
        time_fn: 現在時刻取得関数 (default `time.monotonic`)。test で fake 時計を
            注入するため DI hook として提供。
        skill_name: dispatcher registry の key。default "setup"。
    """

    name = "setup"

    def __init__(
        self,
        stages: list[SetupStage],
        *,
        time_fn: Callable[[], float] = time.monotonic,
        skill_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("stages must be a non-empty list")
        self._stages: tuple[SetupStage, ...] = tuple(stages)
        self._time_fn = time_fn
        if skill_name is not None:
            self.name = skill_name
        self._started_at: Optional[float] = None

    def _on_start(self, params: dict) -> None:
        # start 時刻を記録するだけ。stage 切り替えは step() 内で elapsed から判定。
        self._started_at = self._time_fn()

    def _on_stop(self) -> None:
        self._started_at = None

    def step(self, obs: dict) -> np.ndarray:
        """current stage の pose を返す (defensive copy)。

        `_on_start` が呼ばれていない or `_started_at` が None の場合は先頭 stage
        の pose を返す (safe default)。
        """
        if self._started_at is None:
            return self._stages[0].pose_rad.copy()
        elapsed = self._time_fn() - self._started_at
        stage = self._stage_at(elapsed)
        return stage.pose_rad.copy()

    def _stage_at(self, elapsed_sec: float) -> SetupStage:
        """elapsed_sec 時点で active な stage を返す。全 stage 消化後は最終 stage。"""
        cumulative = 0.0
        for stage in self._stages:
            cumulative += stage.dwell_sec
            if elapsed_sec < cumulative:
                return stage
        return self._stages[-1]

    @property
    def total_dwell_sec(self) -> float:
        """全 stage の dwell 合計 (= max_dwell_sec の実体)。"""
        return sum(stage.dwell_sec for stage in self._stages)

    @property
    def max_dwell_sec(self) -> float:
        """Skill 基底 property の override (Issue #81 Phase 3)。

        全 stage 消化に必要な時間 = `total_dwell_sec` を返す。Orchestrator が
        この値を読んで、setup 完了後 (auto-transition 経由で) move_to_table へ進む。
        """
        return self.total_dwell_sec

    @property
    def stages(self) -> tuple[SetupStage, ...]:
        return self._stages

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        *,
        time_fn: Callable[[], float] = time.monotonic,
        skill_name: Optional[str] = None,
    ) -> "SetupSkill":
        """dict (skill_config.yaml の `skills.setup` セクション) から SetupSkill を構築する。

        期待する構造:

            {
              "stages": [
                {"name": <str>, "dwell_sec": <float>, "pose_rad": {joint: value, ...}},
                ...
              ]
            }

        primary API。YAML I/O は呼び出し元 (entrypoint) に集約する Phase 2b 方針。
        """
        if not isinstance(cfg, dict) or "stages" not in cfg:
            raise ValueError("setup config: 'stages' is required")
        raw_stages = cfg["stages"]
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("setup config: 'stages' must be a non-empty list")

        stages: list[SetupStage] = []
        for i, entry in enumerate(raw_stages):
            if not isinstance(entry, dict):
                raise ValueError(f"setup config: stage[{i}] must be a mapping")
            name = entry.get("name") or f"stage_{i}"
            if "dwell_sec" not in entry:
                raise ValueError(
                    f"setup config: stage {name!r}: 'dwell_sec' is required"
                )
            dwell_sec = float(entry["dwell_sec"])
            pose_rad = densify_pose(
                entry.get("pose_rad", {}), context=f"stage {name!r}"
            )
            stages.append(
                SetupStage(name=name, dwell_sec=dwell_sec, pose_rad=pose_rad)
            )
        return cls(stages, time_fn=time_fn, skill_name=skill_name)

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        time_fn: Callable[[], float] = time.monotonic,
        skill_name: Optional[str] = None,
    ) -> "SetupSkill":
        """YAML path から SetupSkill を構築する (from_config の convenience wrapper)。

        `skills.setup` セクションを抽出して `from_config` に委譲。
        `yaml.safe_load` 使用 (Python object 復元を封じる)。
        """
        text = Path(path).read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: YAML root must be a mapping")
        try:
            setup_cfg = raw["skills"]["setup"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"{path}: missing required section 'skills.setup'"
            ) from exc
        return cls.from_config(setup_cfg, time_fn=time_fn, skill_name=skill_name)
