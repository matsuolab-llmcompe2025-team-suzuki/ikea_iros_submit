"""教師データの関節範囲を出た腕関節だけを内側へ戻す補正 (Issue #141 束 1-4)。

# なぜ要るか

09-09 の実機 11 run のうち 5 run で、腕が URDF 限界に達して 19D 目標がまるごと
保持され (`MotionLimiter` の hold)、腕が止まったまま抜け出せなくなった。hold 中も
policy の生の指令は 2.9〜7.7 deg/s で動き続けていて、入力がほとんど変わらないので
同じ方向の指令が出続ける。

hold の手前では 1 関節が 12〜18 s かけて**教師データの範囲の外**へ流れている
(DR-1)。限界の近くには教師データがほとんど無い (merged_v1 146,873 frame で、
R.elbow は限界から 0.4 rad 以内が 0 frame)。つまり hold に達した時点で model は
学習データの無い領域におり、そこからの復帰を model の出力に任せることはできない。

そこで限界 (hold) を待たず、**関節が教師の範囲を出た時点で**、その関節だけを
範囲の端から `margin_rad` 内側へ戻す。他の関節は policy の指令どおりに通す。
体が動いて状態が変わるので、hold のような「同じ入力 → 同じ出力」の膠着にならない。

# 決めたこと (2026-09-14)

- 範囲の基準は教師の **min / max**。p1 / p99 だと補正が常時かかる
  (09-09 実測: min/max 基準なら範囲外の関節は median 0〜1 個、p1/p99 基準だと 1〜7 個)
- 補正するのは**範囲の外に出た関節だけ**。範囲の内側にいる限り、端の近くでも触らない
- 戻す速さは専用に持たず `MotionLimiter` の速度・加速度の上限に任せる
- 19D 全体の hold (URDF 限界 − 0.03 rad) は最後の安全網として残す。通常はこちらが先に効く
  (教師が URDF の限界まで使った関節は、範囲を hold の限界と交わらせてから使う。
  `within_limits`、Issue #159 B4b-01。交わらせないと「こちらが先に効く」が成り立たない)
- 範囲は skill ごとに教師データから実測して `skill_config.yaml` に持たせる
  (`evaluate/model_evaluation/tools/compute_teacher_joint_range.py`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from inference.desktop.lower_policy.pose_utils import JOINT_NAMES, NUM_ARM_JOINTS

# VlaSkill の 19D layout: waist 3 + arms 14 + hand 2。補正するのは腕 14 だけ。
ARMS_19D = slice(3, 17)


@dataclass(frozen=True)
class TeacherRangeResult:
    """1 tick 分の補正結果。

    Attributes:
        target_19d: 補正後の 19D 絶対目標 (未発火なら入力と同一値)。
        corrected_joints: 補正した関節の short label (`JOINT_NAMES` の名前)。
        max_excursion_rad: 教師の範囲をはみ出していた最大量 [rad] (未発火なら 0.0)。
    """

    target_19d: np.ndarray
    corrected_joints: tuple[str, ...]
    max_excursion_rad: float

    @property
    def bind(self) -> bool:
        return bool(self.corrected_joints)


class TeacherJointRange:
    """教師の範囲を出た腕関節を、端から `margin_rad` 内側へ引き戻す。

    Args:
        lower_rad / upper_rad: 腕 14 関節の教師 min / max [rad] (`JOINT_NAMES` の順)。
        margin_rad: 範囲の端から内側へ入れる量 [rad]。
    """

    def __init__(
        self,
        lower_rad: Any,
        upper_rad: Any,
        *,
        margin_rad: float,
    ) -> None:
        lower = np.asarray(lower_rad, dtype=np.float64)
        upper = np.asarray(upper_rad, dtype=np.float64)
        if lower.shape != (NUM_ARM_JOINTS,) or upper.shape != (NUM_ARM_JOINTS,):
            raise ValueError(
                f"lower_rad / upper_rad must be ({NUM_ARM_JOINTS},), "
                f"got {lower.shape} / {upper.shape}"
            )
        if margin_rad < 0.0:
            raise ValueError(f"margin_rad must be >= 0, got {margin_rad}")
        width = upper - lower
        if np.any(width <= 2.0 * margin_rad):
            narrow = [
                f"{JOINT_NAMES[i]}={width[i]:.3f}"
                for i in np.flatnonzero(width <= 2.0 * margin_rad)
            ]
            raise ValueError(
                f"teacher range narrower than 2 * margin_rad ({margin_rad}): {narrow}"
            )
        self._lower = lower
        self._upper = upper
        self._margin_rad = float(margin_rad)
        # 補正後の目標。範囲外の関節はこの値へ向かう。
        self._pull_low = lower + margin_rad
        self._pull_high = upper - margin_rad

    @property
    def margin_rad(self) -> float:
        return self._margin_rad

    def within_limits(
        self, lower_rad: Any, upper_rad: Any
    ) -> tuple["TeacherJointRange", tuple[str, ...]]:
        """範囲を `[lower_rad, upper_rad]` と交わらせた新しい補正と、削った関節名。

        教師が URDF の限界まで使った関節では、教師の端 = 限界そのもの。MotionLimiter
        (URDF − 0.03) と交わらせないと、(限界 − 0.03, 教師の端] の帯で補正が効かずに
        19D 全体が保持され、範囲を出ても引き戻し先 (端 − margin) がまた拒否される
        (Issue #159 B4b-01)。交わらせれば引き戻し先は限界 − 0.03 − margin になる。
        """
        low = np.asarray(lower_rad, dtype=np.float64)
        high = np.asarray(upper_rad, dtype=np.float64)
        if low.shape != (NUM_ARM_JOINTS,) or high.shape != (NUM_ARM_JOINTS,):
            raise ValueError(
                f"limits must be ({NUM_ARM_JOINTS},), got {low.shape} / {high.shape}"
            )
        lower = np.maximum(self._lower, low)
        upper = np.minimum(self._upper, high)
        narrowed = tuple(
            JOINT_NAMES[i]
            for i in np.flatnonzero((lower != self._lower) | (upper != self._upper))
        )
        return TeacherJointRange(lower, upper, margin_rad=self._margin_rad), narrowed

    def apply(self, target_19d: Any) -> TeacherRangeResult:
        """19D 絶対目標の腕 14 関節に補正をかける。"""
        target = np.asarray(target_19d, dtype=np.float64)
        if target.shape != (19,):
            raise ValueError(f"target_19d must be (19,), got {target.shape}")
        arms = target[ARMS_19D]
        below = arms < self._lower
        above = arms > self._upper
        outside = np.flatnonzero(below | above)
        if outside.size == 0:
            return TeacherRangeResult(
                target_19d=np.asarray(target_19d), corrected_joints=(), max_excursion_rad=0.0
            )
        excursion = np.maximum(self._lower - arms, arms - self._upper)
        corrected = target.copy()
        corrected_arms = corrected[ARMS_19D]
        corrected_arms[below] = self._pull_low[below]
        corrected_arms[above] = self._pull_high[above]
        return TeacherRangeResult(
            target_19d=corrected,
            corrected_joints=tuple(JOINT_NAMES[i] for i in outside),
            max_excursion_rad=float(excursion[outside].max()),
        )


def load_teacher_joint_range_for_skill(
    skill_config: Any, skill_name: str
) -> Optional[TeacherJointRange]:
    """skill_config.yaml から補正を解決。`teacher_joint_range` block が無ければ None。

    `load_progress_monitor_for_skill` と同じ解決方針 (default は持たない)。範囲は
    skill ごとに教師データが違うので、書いてある skill でだけ有効になる。

    Args:
        skill_config: yaml.safe_load(skill_config.yaml) の結果 (top-level dict)。
        skill_name: `skills.<name>` の key。

    Returns:
        TeacherJointRange、または `teacher_joint_range` 未記載なら None。
    """
    skills = skill_config.get("skills") if hasattr(skill_config, "get") else None
    entry = (skills or {}).get(skill_name) or {}
    cfg = entry.get("teacher_joint_range")
    if not isinstance(cfg, dict):
        return None
    unknown = set(cfg) - {"margin_rad", "source", "arm_position_rad"}
    if unknown:
        raise ValueError(
            f"skills.{skill_name}.teacher_joint_range has unknown keys: {sorted(unknown)}"
        )
    ranges = cfg.get("arm_position_rad")
    if not isinstance(ranges, dict):
        raise ValueError(
            f"skills.{skill_name}.teacher_joint_range.arm_position_rad must be a mapping"
        )
    missing = [name for name in JOINT_NAMES if name not in ranges]
    unknown_joints = sorted(set(ranges) - set(JOINT_NAMES))
    if missing or unknown_joints:
        raise ValueError(
            f"skills.{skill_name}.teacher_joint_range.arm_position_rad: "
            f"missing={missing} unknown={unknown_joints}"
        )
    lower, upper = [], []
    for name in JOINT_NAMES:
        pair = ranges[name]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"skills.{skill_name}.teacher_joint_range.arm_position_rad.{name} "
                f"must be [min, max], got {pair!r}"
            )
        low, high = float(pair[0]), float(pair[1])
        if not low < high:
            raise ValueError(
                f"skills.{skill_name}.teacher_joint_range.arm_position_rad.{name}: "
                f"min must be < max, got [{low}, {high}]"
            )
        lower.append(low)
        upper.append(high)
    if "margin_rad" not in cfg:
        raise ValueError(
            f"skills.{skill_name}.teacher_joint_range requires margin_rad"
        )
    return TeacherJointRange(lower, upper, margin_rad=float(cfg["margin_rad"]))
