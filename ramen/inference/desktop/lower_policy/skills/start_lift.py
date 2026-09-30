"""開始待ちで、手先を向きを保ったまま 1 cm ずつ上げ下げする (会場の経路だけ、Issue #188)。

会場 (運営 WBC) では胴の傾きなどで開始の手の高さが変わり、pick では指先が台に接して
脚を上から掴めなかった。開始姿勢は skill_config の ``boundary_profile`` で上げてあるが、
台や脚の置き方は会場ごとに違うので、開始待ちの間に操作者が U / D キーで合わせられるようにする。

- 基準は会場の開始姿勢 (``boundary_profile`` を重ねた後)。そこから真上 (胴の座標) へ
  ``lift - base_lift_m`` だけ動かす腕の関節角を、G1 URDF の IK で出す (向きは保つ)。
- 量は「元の開始姿勢 (学習の中央値) から何 m 上げたか」で持つ (表示と範囲に使う)。
- 変えた量はその run の中だけ効く (R の戻りの後も同じ量)。次の run は ``base_lift_m`` から。

値は skill_config の ``skills.<skill>.start_lift_keys`` (CLAUDE.md: skill の数値は YAML)。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: 開始待ちで受け付けるキー (上げる, 下げる)。
KEY_UP = "u"
KEY_DOWN = "d"
KEYS = (KEY_UP, KEY_DOWN)

# IK の打ち切り (位置 [m] と向き [rad] の許容)
_IK_POSITION_TOLERANCE_M = 1e-4
_IK_ORIENTATION_TOLERANCE_RAD = 1e-3
_IK_ITERATIONS = 100
# URDF の端から残す幅 [rad] (run_skill の目標の限界と同じ)
_LIMIT_MARGIN_RAD = 0.03
_HAND_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
#: ``start_lift_keys.hand`` → 上げ下げする手。both は両手を同じ量ずつ (量は右手で表す)。
_HANDS = {"left": ("left",), "right": ("right",), "both": ("right", "left")}


@dataclass(frozen=True)
class StartLiftConfig:
    """``start_lift_keys`` の中身。量はどれも「元の開始姿勢から上げた量 [m]」。"""

    hand: str
    step_m: float
    base_lift_m: float
    min_lift_m: float
    max_lift_m: float
    joint_speed_rad_s: float

    def __post_init__(self) -> None:
        if self.hand not in _HANDS:
            raise ValueError(
                f"start_lift_keys.hand must be left, right or both, got {self.hand!r}"
            )
        values = (self.step_m, self.base_lift_m, self.min_lift_m, self.max_lift_m)
        if not all(np.isfinite(v) for v in (*values, self.joint_speed_rad_s)):
            raise ValueError("start_lift_keys values must be finite")
        if not 0.0 < self.step_m <= 0.02:
            raise ValueError(
                f"start_lift_keys.step_m must be in (0, 0.02] m, got {self.step_m}"
            )
        if not self.min_lift_m <= self.base_lift_m <= self.max_lift_m <= 0.10:
            raise ValueError(
                "start_lift_keys needs min_lift_m <= base_lift_m <= max_lift_m <= 0.10 m, "
                f"got {self.min_lift_m}, {self.base_lift_m}, {self.max_lift_m}"
            )
        if not 0.0 < self.joint_speed_rad_s <= 0.5:
            raise ValueError(
                f"start_lift_keys.joint_speed_rad_s must be in (0, 0.5], got {self.joint_speed_rad_s}"
            )

    @classmethod
    def from_skill(
        cls, skill_section: dict, skill_name: str
    ) -> Optional["StartLiftConfig"]:
        """skill の設定から読む。``start_lift_keys`` が無ければ None (キーを出さない)。"""
        raw = skill_section.get("start_lift_keys")
        if raw is None:
            return None
        expected = {
            "hand",
            "step_m",
            "base_lift_m",
            "min_lift_m",
            "max_lift_m",
            "joint_speed_rad_s",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(
                f"skills.{skill_name}.start_lift_keys must have exactly {sorted(expected)}"
            )
        return cls(
            hand=str(raw["hand"]),
            **{key: float(raw[key]) for key in sorted(expected - {"hand"})},
        )


def _tool(fk, q29: np.ndarray, hand: str) -> tuple[np.ndarray, np.ndarray]:
    chain, offset = (
        (fk._left_chain, fk._left_offset)
        if hand == "left"
        else (fk._right_chain, fk._right_offset)
    )
    transform = fk._fk_chain(chain, q29)
    return transform[:3, 3] + transform[:3, :3] @ offset, transform[:3, :3]


def _pose_error(fk, q29, hand, target_p, target_r) -> np.ndarray:
    p, r = _tool(fk, q29, hand)
    w = target_r.T @ r
    rotation = np.array([w[2, 1] - w[1, 2], w[0, 2] - w[2, 0], w[1, 0] - w[0, 1]]) / 2.0
    return np.concatenate(
        [(p - target_p) * 10.0, rotation]
    )  # 1 cm と 0.1 rad を同じ重み


def lift_arm_pose(arm14: Sequence[float], hand: str, dz_m: float, fk) -> np.ndarray:
    """手先を向きを保ったまま、胴の座標で真上へ ``dz_m`` 動かした腕 14 関節。

    減衰つき最小二乗の IK (数値微分)。7 関節の余りは小さく動く方へ解く。解けなければ
    ValueError (呼出側は上げ下げを断る)。
    """
    base = np.asarray(arm14, dtype=np.float64).reshape(-1)
    if base.shape != (14,) or not np.isfinite(base).all():
        raise ValueError("arm pose must be finite [14]")
    sl = _HAND_SLICES[hand]
    offset = 15 + sl.start
    q = np.zeros(29)
    q[15:29] = base
    p0, r0 = _tool(fk, q, hand)
    target = p0 + np.array([0.0, 0.0, float(dz_m)])
    for _ in range(_IK_ITERATIONS):
        err = _pose_error(fk, q, hand, target, r0)
        if (
            np.linalg.norm(err[:3]) / 10.0 < _IK_POSITION_TOLERANCE_M
            and np.linalg.norm(err[3:]) < _IK_ORIENTATION_TOLERANCE_RAD
        ):
            break
        jacobian = np.zeros((6, 7))
        for k in range(7):
            dq = q.copy()
            dq[offset + k] += 1e-5
            jacobian[:, k] = (_pose_error(fk, dq, hand, target, r0) - err) / 1e-5
        step = np.linalg.solve(
            jacobian.T @ jacobian + 1e-4 * np.eye(7), -jacobian.T @ err
        )
        q[offset : offset + 7] += step
    err = _pose_error(fk, q, hand, target, r0)
    if (
        np.linalg.norm(err[:3]) / 10.0 >= 10 * _IK_POSITION_TOLERANCE_M
        or np.linalg.norm(err[3:]) >= 10 * _IK_ORIENTATION_TOLERANCE_RAD
    ):
        raise ValueError(
            f"cannot move the {hand} hand {dz_m * 100:+.1f} cm keeping its orientation"
        )
    return q[15:29].copy()


class StartLiftAdjuster:
    """開始待ちの目標の腕 (14 関節) を、U / D キーで 1 段ずつ上げ下げする。

    ``target()`` は制御の thread、``adjust()`` はキーを待つ thread から呼ばれるので lock で守る。
    """

    keys = KEYS

    def __init__(
        self,
        base_arm14: Sequence[float],
        config: StartLiftConfig,
        *,
        fk=None,
        teacher_range: Optional[dict] = None,
    ) -> None:
        from inference.desktop.lower_policy.actuators.g1_arm_sdk import (
            G1_ARM_POSITION_LOWER_RAD,
            G1_ARM_POSITION_UPPER_RAD,
        )
        from inference.desktop.lower_policy.initial_pose import ARM_JOINT_ORDER

        if fk is None:
            from inference.desktop.perception.g1_urdf_fk import G1WristFK

            fk = G1WristFK.from_urdf()
        self._fk = fk
        self._config = config
        self._base = np.asarray(base_arm14, dtype=np.float64).reshape(-1).copy()
        if self._base.shape != (14,) or not np.isfinite(self._base).all():
            raise ValueError("start lift base pose must be finite [14]")
        lower = (
            np.asarray(G1_ARM_POSITION_LOWER_RAD, dtype=np.float64) + _LIMIT_MARGIN_RAD
        )
        upper = (
            np.asarray(G1_ARM_POSITION_UPPER_RAD, dtype=np.float64) - _LIMIT_MARGIN_RAD
        )
        if teacher_range:
            for index, name in enumerate(ARM_JOINT_ORDER):
                if name in teacher_range:
                    low, high = teacher_range[name]
                    lower[index] = max(lower[index], float(low))
                    upper[index] = min(upper[index], float(high))
        self._lower, self._upper = lower, upper
        self._lift = config.base_lift_m
        self._target = self._base.copy()
        self._lock = threading.Lock()

    @property
    def lift_m(self) -> float:
        with self._lock:
            return self._lift

    def target(self) -> np.ndarray:
        """今の開始待ちの目標の腕 14 関節。"""
        with self._lock:
            return self._target.copy()

    def status(self) -> str:
        hand = {"right": "右手", "left": "左手", "both": "両手（右手の量）"}[self._config.hand]
        return (
            f"{hand} +{self.lift_m * 100:.1f} cm（元の開始姿勢から。U で "
            f"{self._config.step_m * 100:.0f} cm 上げる・D で下げる、"
            f"{self._config.min_lift_m * 100:.0f}〜{self._config.max_lift_m * 100:.0f} cm）"
        )

    def adjust(self, key: str) -> str:
        """U / D を 1 回。新しい目標を作れたら切り替え、今の量の 1 行を返す。

        範囲の端・IK が解けない・関節の範囲を越えるときは目標を変えずに理由を返す。
        """
        direction = {KEY_UP: 1.0, KEY_DOWN: -1.0}.get(key)
        if direction is None:
            raise ValueError(f"unknown start lift key {key!r}")
        config = self._config
        with self._lock:
            current = self._lift
        # 端では残りの分だけ動かし、ちょうど端で止める (1 段の刻みが端に揃わなくても端まで行ける)
        lift = round(
            min(max(current + direction * config.step_m, config.min_lift_m), config.max_lift_m), 6
        )
        if abs(lift - current) < 1e-9:
            return f"[start-lift] 範囲の端です。{self.status()}"
        try:
            arm = self._base
            for hand in _HANDS[config.hand]:
                # 腰は動かさないので、左右の腕は別々に解ける (片方の 7 関節だけが変わる)
                arm = lift_arm_pose(arm, hand, lift - config.base_lift_m, self._fk)
        except ValueError as exc:
            return f"[start-lift] 動かせません ({exc})。{self.status()}"
        outside = (arm < self._lower) | (arm > self._upper)
        if outside.any():
            return f"[start-lift] 関節の範囲を越えるので動かしません。{self.status()}"
        with self._lock:
            self._lift = lift
            self._target = arm
        return f"[start-lift] {self.status()}"

    def ramp(self, held: np.ndarray, dt_s: float) -> np.ndarray:
        """保持している目標を、今の目標へ関節の速さを抑えて近づける (1 tick 分)。"""
        goal = self.target()
        step = self._config.joint_speed_rad_s * max(0.0, min(float(dt_s), 0.1))
        return held + np.clip(goal - held, -step, step)
