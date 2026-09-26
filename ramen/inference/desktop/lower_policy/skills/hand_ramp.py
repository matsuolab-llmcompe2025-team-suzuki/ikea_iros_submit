"""Dex1 の開度を動かすための ramp と到達判定 (Issue #141 束 1-8 / D2)。

評価経路にしか無かった「手を開く → 掴む幅へ閉じる」手順を、評価と本番の両方から
使えるようにここへ移した (元: `evaluate/model_evaluation/common/safe_return.py` と
`runners/run_skill.py`)。skill の形にしたものは `hand_pre_motion.py`。

到達の判定は 3 通り:
    - `target`: 目標との差が `tolerance_rad` 以内 (開く動きと、物を持たない移動)
    - `closing_contact`: 閉じる動きで、指令したストロークの 80% 以上進んだところで
      安定した (脚を掴んだ = 物に当たって止まった)。力センサが無いので、
      「動かなくなった位置」で接触を判定する
    - `holding_contact`: 始めから物を握っていた側 (`holding`) が、そのまま安定した。
      握った手は 1 mm も閉じられないので、80% の条件を掛けない (Issue #159 B4a-04)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from inference.desktop.lower_policy.actuators.hand import HAND_GRIP_MAX, HAND_GRIP_MIN

# 接触判定の既定値 (評価経路の実績値。skill 側から上書きできる)
CONTACT_TOLERANCE_RAD = 0.35
MAXIMUM_CONTACT_RESIDUAL_RAD = 0.75
MINIMUM_PROGRESS_FRACTION = 0.80
STABLE_SPAN_RAD = 0.03
STABLE_HISTORY_LEN = 10


def resolve_hand_opening_rad(skill_config: dict, *, action_sink: str = "sdk") -> float:
    """Resolve preparation/release opening in physical radians, not dataset units.

    Only the open/release target is capped for the selected transport.  Dataset
    frame-zero and policy hand targets must retain their original calibration.
    """
    opening = float((skill_config.get("hand_pre_motion") or {}).get("open_rad", HAND_GRIP_MAX))
    if not math.isfinite(opening) or not 0.0 < opening <= HAND_GRIP_MAX:
        raise ValueError(f"hand_pre_motion.open_rad must be in (0, {HAND_GRIP_MAX}], got {opening}")
    if action_sink == "sdk":
        return opening
    if action_sink == "boundary":
        from inference.desktop.lower_policy.policies.taskspace_adapter import (
            DEX1_BOUNDARY_FULL_OPEN_RAD,
        )

        return min(opening, DEX1_BOUNDARY_FULL_OPEN_RAD)
    raise ValueError(f"unknown hand action sink: {action_sink!r}")


def build_hand_target_ramp(
    measured_hand_rad: Sequence[float],
    target_hand_rad: Sequence[float],
    *,
    command_hz: float = 30.0,
    velocity_limit_rad_s: float = 1.5,
) -> tuple[np.ndarray, ...]:
    """今の開度から目標までの指令列を作る (速度の上限で刻む)。

    指令の位置だけを変える。接触の推定もしないし、目標を黙って書き換えることもしない。
    """
    measured = np.asarray(measured_hand_rad, dtype=np.float64)
    target = np.asarray(target_hand_rad, dtype=np.float64)
    if measured.shape != (2,) or not np.isfinite(measured).all():
        raise ValueError("measured Dex1 pose must be finite 2-D")
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("target Dex1 pose must be finite 2-D")
    if command_hz <= 0.0 or velocity_limit_rad_s <= 0.0:
        raise ValueError("hand command rate and velocity limit must be positive")
    start = np.clip(measured, HAND_GRIP_MIN, HAND_GRIP_MAX)
    target = np.clip(target, HAND_GRIP_MIN, HAND_GRIP_MAX)
    maximum_delta = float(np.max(np.abs(target - start)))
    steps = max(1, int(math.ceil(maximum_delta * command_hz / velocity_limit_rad_s)))
    return tuple(
        start + (index / steps) * (target - start) for index in range(1, steps + 1)
    )


def build_hand_open_ramp(
    measured_hand_rad: Sequence[float],
    *,
    command_hz: float = 30.0,
    velocity_limit_rad_s: float = 1.5,
    open_rad: float = HAND_GRIP_MAX,
) -> tuple[np.ndarray, ...]:
    """全開までの指令列。"""
    opening = resolve_hand_opening_rad({"hand_pre_motion": {"open_rad": open_rad}})
    return build_hand_target_ramp(
        measured_hand_rad,
        (opening, opening),
        command_hz=command_hz,
        velocity_limit_rad_s=velocity_limit_rad_s,
    )


def hand_target_completion_mode(
    *,
    start: np.ndarray,
    target: np.ndarray,
    measured_history: list[np.ndarray],
    tolerance_rad: float,
    allow_closing_contact: bool,
    contact_tolerance_rad: float = CONTACT_TOLERANCE_RAD,
    maximum_contact_residual_rad: float = MAXIMUM_CONTACT_RESIDUAL_RAD,
    minimum_progress_fraction: float = MINIMUM_PROGRESS_FRACTION,
    stable_span_rad: float = STABLE_SPAN_RAD,
    holding: Optional[Sequence[bool]] = None,
    active: Optional[Sequence[bool]] = None,
) -> Optional[str]:
    """到達したか、物に当たって止まったかを判定する。

    Args:
        holding: 左右それぞれ、始めから物を握っていたか。True の側は進んだ量を
            問わず、残差は `maximum_contact_residual_rad` まで許す。
        active: 左右それぞれ、到達を見るか。False の側は動かしていないので見ない
            (握っている手だけ離すとき、Issue #159 T3)。

    Returns:
        "target" (目標に入った) / "closing_contact" (閉じる途中で接触して安定した) /
        "holding_contact" (握っていた手がそのまま安定した) / None (まだ)。
    """
    if not measured_history:
        return None
    measured = np.asarray(measured_history[-1], dtype=np.float64)
    residual = measured - target
    watched = (
        np.ones(2, dtype=bool)
        if active is None
        else np.asarray(active, dtype=bool).reshape(2)
    )
    if np.all((np.abs(residual) <= tolerance_rad) | ~watched):
        return "target"
    if not allow_closing_contact or len(measured_history) < STABLE_HISTORY_LEN:
        return None
    history = np.asarray(measured_history[-STABLE_HISTORY_LEN:], dtype=np.float64)
    if history.shape != (STABLE_HISTORY_LEN, 2) or not np.isfinite(history).all():
        return None
    if np.max(np.ptp(history, axis=0)) > stable_span_rad:
        return None

    stroke = start - target
    progress = np.divide(
        start - measured,
        stroke,
        out=np.zeros_like(stroke),
        where=np.abs(stroke) > tolerance_rad,
    )
    per_side_ok = np.abs(residual) <= tolerance_rad
    held = (
        np.zeros(2, dtype=bool)
        if holding is None
        else np.asarray(holding, dtype=bool).reshape(2)
    )
    # 脚に当たると、指令した目標より開いた位置で止まる。許す残差はストロークに比例
    # させ、0.75 rad で頭打ちにする。握っていた側は物の太さで止まっているので上限まで。
    contact_residual_limit = np.where(
        held,
        maximum_contact_residual_rad,
        np.minimum(
            maximum_contact_residual_rad,
            np.maximum(
                contact_tolerance_rad,
                np.abs(stroke) * (1.0 - minimum_progress_fraction),
            ),
        ),
    )
    closing_contact = (
        (stroke > tolerance_rad)
        & (residual >= -tolerance_rad)
        & (residual <= contact_residual_limit)
        & ((progress >= minimum_progress_fraction) | held)
    )
    if np.all(per_side_ok | closing_contact) and np.any(closing_contact):
        return "holding_contact" if np.any(closing_contact & held) else "closing_contact"
    return None
