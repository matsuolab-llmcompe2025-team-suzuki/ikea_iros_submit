"""運営 WBC で腕の遅れ・押し負けを、送り方で補う (boundary の joint lane、Issue #188)。

背景 (09-29 の会場の log と、自前の実機評価 = Regular Mode + ``rt/arm_sdk`` の比較):

- 運営 adapter は chunk を受け取るたびに 1 行目を**実測の腕**に合わせ直し、1 行 (50 ms)
  ごとに ``max_joint_vel / chunk_hz`` (既定 0.05 rad) しか進めない
  (``wbc_driver.py:_step_clamp``)。
- 私たちは動いている間 ``publish_period_s`` (0.1 s) おきに送り直すので、WBC の目標は
  実測より約 0.1 rad しか先に出ない。WBC の腕は柔らかい (肘 kp 40、実機はトルク 0 の PD) ので、
  台に押し返されると取り戻せず (台を回すときの右肘 0.15 rad)、遅れも 270〜390 ms に膨らむ
  (自前は 80〜110 ms)。

補い方は 4 つ (値は ``skill_config.yaml`` の ``boundary_arm_tracking``。0 ならその補いを使わない):

- 押している間の送る間隔: 最後に送った指令 (重力の垂れ補正を除いた、WBC が着くはずの腕) と
  実測の差が ``blocked_gap_rad`` を超え、その関節が ``blocked_still_rad_s`` より遅い間は、
  送る間隔を ``blocked_period_s`` に延ばす。
  送り直さない間は adapter が目標まで進み切るので、目標が実測より先に出て押す力が出る。
- 押し込み補正: 押している関節にだけ、``push_gain`` × (policy の指令 − 実測) を足す
  (``push_max_rad`` まで)。台が動いて差が縮めば足す量も縮む (積み上げない)。足す量は
  1 秒あたり ``_PUSH_RATE_RAD_S`` までしか変えない。
- 先読み: 指令の速さ (``lead_window_s`` でならす) × ``lead_horizon_s`` を指令に足す
  (関節ごとに ``lead_max_rad`` まで)。過去の指令だけから速さを出すので、向きの変わり目では
  行き過ぎる (09-29 の log の模擬で pick のずれが増えた)。
- 先の予定: 腕だけ、policy の予定の ``preview_s`` 先を今送る (手・腰は今のまま)。過去から
  推し量る先読みと違い、policy が立てた予定そのものなので向きの変わり目でも行き過ぎない。
  差し替えるのは ``VlaSkill`` (``set_arm_preview_steps``)。ここは値を持つだけ。09-29 の log の
  模擬 (後で実際に実行された指令を予定とみなした上限) で、0.2 s で pick の遅れ 340→130 ms、
  台を回す 250→40 ms。

どれも adapter の上限 (1 rad/s) の内側で働き (速さの上限は変えない)、URDF の端は越えない。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: ``--arm-tuning`` で選べる束の名前 (``skill_config.yaml`` の ``boundary_arm_tracking.presets``)。
ARM_TUNING_PRESETS = ("today", "standard", "lead")

# 指令がこれより長く来なければ、ならした速さを捨てる [s] (skill の切り替え・準備動作の後)。
_HISTORY_GAP_RESET_S = 0.3
# 速さを出すのに要る最短の時間幅 [s] (起動直後の 1〜2 tick の差分で先読みしない)。
_MIN_VELOCITY_SPAN_S = 0.05
# 実測の速さをならす時定数 [s]。
_MEASURED_SPEED_TAU_S = 0.1
# 押し込み補正の量を変える速さの上限 [rad/s] (急に押し込んだり抜いたりしない)。
_PUSH_RATE_RAD_S = 0.5
# 補いで URDF の端ちょうどまで押さない幅 [rad] (重力の垂れ補正と同じ)。
_LIMIT_MARGIN_RAD = 0.005

_KEYS = (
    "lead_horizon_s",
    "lead_window_s",
    "lead_max_rad",
    "blocked_period_s",
    "blocked_gap_rad",
    "blocked_still_rad_s",
    "push_gain",
    "push_max_rad",
    "preview_s",
)


@dataclass(frozen=True)
class ArmTrackingConfig:
    """``boundary_arm_tracking`` の 1 つの束。0 の項目はその補いを使わない。"""

    lead_horizon_s: float
    lead_window_s: float
    lead_max_rad: float
    blocked_period_s: float
    blocked_gap_rad: float
    blocked_still_rad_s: float
    push_gain: float
    push_max_rad: float
    # 腕だけ policy の予定の何秒先を送るか (0 = 今の step)。
    preview_s: float = 0.0

    def __post_init__(self) -> None:
        for key in _KEYS:
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"boundary_arm_tracking.{key} must be finite and >= 0, got {value}"
                )
        limits = {
            "lead_horizon_s": 0.5,
            "lead_max_rad": 0.2,
            "blocked_period_s": 1.0,
            "push_gain": 2.0,
            "push_max_rad": 0.2,
            "preview_s": 0.5,
        }
        for key, upper in limits.items():
            if getattr(self, key) > upper:
                raise ValueError(f"{key} must be <= {upper}, got {getattr(self, key)}")
        if not 0.05 <= self.lead_window_s <= 1.0:
            raise ValueError(
                f"lead_window_s must be in [0.05, 1.0] s, got {self.lead_window_s}"
            )
        if (self.uses_blocked_period or self.uses_push) and (
            self.blocked_gap_rad <= 0.0 or self.blocked_still_rad_s <= 0.0
        ):
            raise ValueError(
                "blocked_gap_rad and blocked_still_rad_s must be > 0 when the blocked "
                "period or the push is used"
            )

    @property
    def uses_lead(self) -> bool:
        return self.lead_horizon_s > 0.0 and self.lead_max_rad > 0.0

    @property
    def uses_blocked_period(self) -> bool:
        return self.blocked_period_s > 0.0

    @property
    def uses_push(self) -> bool:
        return self.push_gain > 0.0 and self.push_max_rad > 0.0

    def preview_steps(self, control_hz: float) -> int:
        """``preview_s`` を制御の tick 数に直す (0 = 先の予定を送らない)。"""
        if not np.isfinite(control_hz) or control_hz <= 0.0:
            raise ValueError(f"control_hz must be > 0, got {control_hz}")
        return int(round(self.preview_s * control_hz))

    @classmethod
    def from_config(
        cls,
        skill_config: dict,
        preset: Optional[str] = None,
        *,
        lead_horizon_s: Optional[float] = None,
        blocked_period_s: Optional[float] = None,
        push_max_rad: Optional[float] = None,
        preview_s: Optional[float] = None,
    ) -> tuple["ArmTrackingConfig", str]:
        """``skill_config.yaml`` の束を読む。(束, 使った束の名前) を返す。

        ``preset`` を省くと ``default_preset``。``lead_horizon_s`` / ``blocked_period_s`` /
        ``push_max_rad`` / ``preview_s`` は会場で合わせるための上書き (None なら束の値)。
        """
        section = skill_config.get("boundary_arm_tracking")
        if not isinstance(section, dict):
            raise ValueError("skill_config.boundary_arm_tracking is missing")
        unknown = sorted(set(section) - {"default_preset", "presets"})
        if unknown:
            raise ValueError(f"boundary_arm_tracking has unknown keys: {unknown}")
        presets = section.get("presets")
        if not isinstance(presets, dict):
            raise ValueError("boundary_arm_tracking.presets must be a mapping")
        missing = [name for name in ARM_TUNING_PRESETS if name not in presets]
        if missing:
            raise ValueError(f"boundary_arm_tracking.presets is missing {missing}")
        name = preset if preset is not None else section.get("default_preset")
        if name not in presets:
            raise ValueError(
                f"unknown arm tracking preset {name!r} (choose from {sorted(presets)})"
            )
        values = presets[name]
        if not isinstance(values, dict):
            raise ValueError(f"boundary_arm_tracking.presets.{name} must be a mapping")
        bad = sorted(set(values) ^ set(_KEYS))
        if bad:
            raise ValueError(
                f"boundary_arm_tracking.presets.{name} has missing/unknown keys: {bad}"
            )
        fields = {key: float(values[key]) for key in _KEYS}
        for key, override in (
            ("lead_horizon_s", lead_horizon_s),
            ("blocked_period_s", blocked_period_s),
            ("push_max_rad", push_max_rad),
            ("preview_s", preview_s),
        ):
            if override is not None:
                fields[key] = float(override)
        return cls(**fields), str(name)

    def describe(self) -> str:
        parts = [
            f"blocked period {self.blocked_period_s:g}s "
            f"(gap > {self.blocked_gap_rad:g}rad and speed < {self.blocked_still_rad_s:g}rad/s)"
            if self.uses_blocked_period
            else "blocked period off",
            f"push {self.push_gain:g}x gap (max {self.push_max_rad:g}rad)"
            if self.uses_push
            else "push off",
            f"lead {self.lead_horizon_s:g}s (window {self.lead_window_s:g}s, max {self.lead_max_rad:g}rad)"
            if self.uses_lead
            else "lead off",
            f"policy plan preview {self.preview_s:g}s (arms only)"
            if self.preview_s > 0.0
            else "preview off",
        ]
        return ", ".join(parts)


class ArmTrackingAssist:
    """1 本の publisher に付く状態 (指令の履歴・実測の速さ・押している関節・押し込みの量)。

    ``BoundaryActionSink`` が tick ごとに ``lead`` と ``update_blocked`` を呼ぶ
    (送らない tick も呼んで、速さの履歴を切らさない)。
    """

    def __init__(self, config: ArmTrackingConfig, *, name: str = "custom") -> None:
        from inference.desktop.lower_policy.actuators.g1_arm_sdk import (
            G1_ARM_POSITION_LOWER_RAD,
            G1_ARM_POSITION_UPPER_RAD,
        )

        self._config = config
        self._name = name
        self._lower = (
            np.asarray(G1_ARM_POSITION_LOWER_RAD, dtype=np.float64) + _LIMIT_MARGIN_RAD
        )
        self._upper = (
            np.asarray(G1_ARM_POSITION_UPPER_RAD, dtype=np.float64) - _LIMIT_MARGIN_RAD
        )
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._measured_prev: Optional[tuple[float, np.ndarray]] = None
        self._measured_speed = np.zeros(14, dtype=np.float64)
        self._blocked_mask = np.zeros(14, dtype=bool)
        self._push = np.zeros(14, dtype=np.float64)
        self._push_updated_at: Optional[float] = None

    @property
    def config(self) -> ArmTrackingConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._name

    @property
    def blocked(self) -> bool:
        """どれかの関節が押している (台などに当たって止まっている) か。"""
        return bool(self._blocked_mask.any())

    @property
    def push(self) -> np.ndarray:
        """今足している押し込みの量 14-D [rad]。"""
        return self._push.copy()

    def describe(self) -> str:
        return f"preset={self._name}: {self._config.describe()}"

    def reset(self) -> None:
        """準備動作 (goto) の後など、指令の流れが途切れたとき。"""
        self._history.clear()
        self._measured_prev = None
        self._measured_speed[:] = 0.0
        self._blocked_mask[:] = False
        self._push[:] = 0.0
        self._push_updated_at = None

    def lead(self, intent14: Sequence[float], now: float) -> np.ndarray:
        """補い (先読み + 押し込み) を足した腕 14-D。どちらも使わない束なら intent のまま。

        policy の値が既に URDF の範囲の外の関節には足さない (範囲外の意図は、従来どおり
        運営 adapter が寄せて数える)。
        """
        intent = np.asarray(intent14, dtype=np.float64).reshape(-1)
        if intent.shape != (14,) or not np.all(np.isfinite(intent)):
            raise ValueError("arm tracking input must be finite arms[14]")
        extra = np.zeros(14, dtype=np.float64)
        if self._config.uses_lead:
            extra += self._velocity_lead(intent, now)
        if self._config.uses_push:
            extra += self._push
        if not extra.any():
            return intent.copy()
        inside = (intent >= self._lower) & (intent <= self._upper)
        return np.where(
            inside, np.clip(intent + extra, self._lower, self._upper), intent
        )

    def _velocity_lead(self, intent: np.ndarray, now: float) -> np.ndarray:
        if self._history and now - self._history[-1][0] > _HISTORY_GAP_RESET_S:
            self._history.clear()
        self._history.append((float(now), intent.copy()))
        window = self._config.lead_window_s
        while len(self._history) > 2 and now - self._history[1][0] >= window:
            self._history.popleft()
        t0, first = self._history[0]
        span = now - t0
        if span < _MIN_VELOCITY_SPAN_S:
            return np.zeros(14, dtype=np.float64)
        velocity = (intent - first) / span
        return np.clip(
            velocity * self._config.lead_horizon_s,
            -self._config.lead_max_rad,
            self._config.lead_max_rad,
        )

    def update_blocked(
        self,
        sent14: Optional[Sequence[float]],
        measured14: Sequence[float],
        now: float,
        *,
        stale: bool = False,
        intent14: Optional[Sequence[float]] = None,
    ) -> bool:
        """押している関節を更新し、押し込みの量を動かす。どれかが押していれば True。

        ``sent14`` は最後に送った腕のうち WBC が着くはずの腕 (重力の垂れ補正を除く。補正の分は
        垂れで打ち消されるので、差に数えると止まっている関節まで押していることになる)、``measured14`` は実測、
        ``intent14`` は今の policy の指令 (押し込みの向きと量に使う)。古い実測 (``stale``)
        では判定を変えない。
        """
        config = self._config
        if not (config.uses_blocked_period or config.uses_push):
            return False
        measured = np.asarray(measured14, dtype=np.float64).reshape(-1)
        if stale or measured.shape != (14,) or not np.all(np.isfinite(measured)):
            return self.blocked
        if self._measured_prev is not None:
            t_prev, q_prev = self._measured_prev
            dt = now - t_prev
            if dt > _HISTORY_GAP_RESET_S:
                self._measured_speed[:] = 0.0
            elif dt > 0.0:
                speed = np.abs(measured - q_prev) / dt
                alpha = min(1.0, dt / _MEASURED_SPEED_TAU_S)
                self._measured_speed += alpha * (speed - self._measured_speed)
        self._measured_prev = (float(now), measured.copy())
        if sent14 is None:
            self._blocked_mask[:] = False
        else:
            gap = np.abs(np.asarray(sent14, dtype=np.float64).reshape(-1) - measured)
            self._blocked_mask = (gap > config.blocked_gap_rad) & (
                self._measured_speed < config.blocked_still_rad_s
            )
        if config.uses_push:
            self._update_push(measured, intent14, now)
        return self.blocked

    def _update_push(
        self, measured: np.ndarray, intent14: Optional[Sequence[float]], now: float
    ) -> None:
        target = np.zeros(14, dtype=np.float64)
        if intent14 is not None:
            intent = np.asarray(intent14, dtype=np.float64).reshape(-1)
            if intent.shape == (14,) and np.all(np.isfinite(intent)):
                target = np.where(
                    self._blocked_mask,
                    np.clip(
                        self._config.push_gain * (intent - measured),
                        -self._config.push_max_rad,
                        self._config.push_max_rad,
                    ),
                    0.0,
                )
        dt = (
            0.0
            if self._push_updated_at is None
            else min(max(now - self._push_updated_at, 0.0), _HISTORY_GAP_RESET_S)
        )
        step = _PUSH_RATE_RAD_S * dt
        self._push += np.clip(target - self._push, -step, step)
        self._push_updated_at = float(now)

    def publish_period(self, base_period_s: float) -> float:
        """今使う送る間隔。押している間は ``blocked_period_s`` まで延ばす。"""
        if self.blocked and self._config.uses_blocked_period:
            return max(float(base_period_s), self._config.blocked_period_s)
        return float(base_period_s)
