"""Per-tick trajectory smoothing for 19-D upper-body targets (Phase 3、Issue #128).

Ported from issue-70-flip-table-data-augmentation branch
(`model/subtask_policy_training/gr00t/temporal_ensemble.py:UpperBodySafetyLimiter`,
`PHYSICAL_ACTION_DIM=16` 固定)、我々の VlaSkill 19D contract
(waist3 + arms14 + hand2) に合わせて **dim=19 で per-dim envelope を構成**。

# 3 段適用 pipeline

1. **Position clip**: `lower <= target <= upper` (無効値検出 + hardware range 保護、
   本 phase では実装 stub、Layer 2 で最終 clamp されるので緩めに設定可)
2. **Velocity limit**: `|target - reference| / dt <= max_velocity` (per-dim)
3. **Acceleration limit**: `|velocity - prev_velocity| / dt <= max_acceleration`

**reference (base)**:
- 初回 apply: `measured` (motor 実位置)
- 2 回目以降: 前回 apply の出力 (trajectory reference)

これは **trajectory smoother** (前 target を基準に次 target を制限)、Actuator の
`_rate_limited_target` (motor 実位置を基準に publish target を制限) と役割が
違う。両方併用 = 2 段防御 (詳細は motion_limits.py の docstring)。

# 使い方

    limiter = MotionLimiter(
        dim=19,
        arm_slice=slice(0, 17),     # waist3 + arm14 (upper joint 全て)
        hand_slice=slice(17, 19),   # dex1 L/R
        arm_velocity=2.0,
        arm_acceleration=10.0,
        hand_velocity=6.0,
        hand_acceleration=30.0,
        control_hz=30.0,
    )
    # skill 開始時:
    limiter.reset()
    # per-tick:
    safe_target = limiter.apply(target=chunk[0], measured=current_19d)

`measured` は 19D 実測 (waist=joint_state[12:15], arm=joint_state[15:29],
hand=hand_state)。dispatch_waist=false の場合でも 19D で apply、waist slice の
出力は下流で捨てられる (limiter 側は責務外)。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class MotionLimiter:
    """Position clip → velocity limit → acceleration limit の 3 段適用。

    per-dim envelope (arm/waist と hand で違う値) を持ち、per-tick で 19D target を
    physically-reachable trajectory に落とす。

    Attributes:
        dim: 対象次元数 (VlaSkill 契約なら 19)。
        control_hz: 適用 frequency [Hz]。VlaSkill.step が 30Hz で呼ぶなら 30。
    """

    def __init__(
        self,
        *,
        dim: int,
        arm_slice: slice,
        hand_slice: slice,
        arm_velocity: float,
        arm_acceleration: float,
        hand_velocity: float,
        hand_acceleration: float,
        control_hz: float,
        lower: Sequence[float] | None = None,
        upper: Sequence[float] | None = None,
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be positive int, got {dim!r}")
        if control_hz <= 0 or not math.isfinite(control_hz):
            raise ValueError(f"control_hz must be positive finite, got {control_hz!r}")
        for name, value in (
            ("arm_velocity", arm_velocity),
            ("arm_acceleration", arm_acceleration),
            ("hand_velocity", hand_velocity),
            ("hand_acceleration", hand_acceleration),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive finite, got {value!r}")

        self.dim = int(dim)
        self.control_hz = float(control_hz)
        self._arm_slice = arm_slice
        self._hand_slice = hand_slice

        # per-dim envelope を build。arm slice (=waist+arm) は arm limits、
        # hand slice は hand limits。issue-70 は 16D 固定 vector を外部から渡す
        # 設計、我々は skill 別 4 スカラー config から per-dim vector 構築する
        # ほうが YAML 記述量が少なく readable。
        self._max_velocity = np.full(self.dim, hand_velocity, dtype=np.float64)
        self._max_velocity[arm_slice] = arm_velocity  # waist+arm を上書き
        self._max_acceleration = np.full(self.dim, hand_acceleration, dtype=np.float64)
        self._max_acceleration[arm_slice] = arm_acceleration

        # Position clip (optional、default = clip 無効化した超保守的 ±100 rad)。
        # 具体的な G1 joint 範囲は Layer 2 (ArmSafetyLimits.joint_max_abs=1.5)
        # + hand actuator HAND_GRIP_MAX=5.4 で最終 clamp されるため、Layer 1 は
        # 明示指定なければ実質 no-op で通過させる (safe default)。
        if lower is None:
            self._lower = np.full(self.dim, -100.0, dtype=np.float64)
        else:
            self._lower = self._require_dim("lower", lower)
        if upper is None:
            self._upper = np.full(self.dim, 100.0, dtype=np.float64)
        else:
            self._upper = self._require_dim("upper", upper)
        if np.any(self._lower > self._upper):
            raise ValueError("lower must be <= upper element-wise")

        self._previous_target: np.ndarray | None = None
        self._previous_velocity = np.zeros(self.dim, dtype=np.float64)

    def _require_dim(self, name: str, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.dim,):
            raise ValueError(
                f"{name} must have shape ({self.dim},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return array

    def reset(self) -> None:
        """skill 遷移 / episode 開始時に呼ぶ (VlaSkill._on_start から)。

        _previous_target を None に戻す → 次 apply は measured を reference と
        する (前 skill の target trajectory を引きずらない)。
        """
        self._previous_target = None
        self._previous_velocity.fill(0.0)

    def apply(
        self,
        *,
        target: Sequence[float],
        measured: Sequence[float],
    ) -> np.ndarray:
        """1 target を smooth した 19D を返す。

        Args:
            target: (dim,) float、Policy が出す raw target (blended)。
            measured: (dim,) float、実測 state。初回 apply の reference に使う
                (以降は _previous_target が優先)。

        Returns:
            (dim,) float64、position/velocity/acceleration の全 envelope を通した
            safe target。
        """
        requested = np.clip(
            self._require_dim("target", target), self._lower, self._upper
        )
        measured_array = self._require_dim("measured", measured)
        reference = (
            measured_array if self._previous_target is None else self._previous_target
        )
        dt = 1.0 / self.control_hz

        # velocity clip: (target - reference) / dt を ±max_velocity で clamp
        desired_velocity = np.clip(
            (requested - reference) / dt,
            -self._max_velocity,
            self._max_velocity,
        )
        # acceleration clip: 前 tick velocity からの delta を ±max_acc*dt で clamp
        velocity_delta = np.clip(
            desired_velocity - self._previous_velocity,
            -self._max_acceleration * dt,
            self._max_acceleration * dt,
        )
        velocity = self._previous_velocity + velocity_delta
        safe = np.clip(reference + velocity * dt, self._lower, self._upper)

        # state update (次 tick 用)
        self._previous_target = safe.copy()
        self._previous_velocity = velocity
        return safe.copy()
