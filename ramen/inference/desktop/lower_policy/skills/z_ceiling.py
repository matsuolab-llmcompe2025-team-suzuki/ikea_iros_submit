"""左手先の高さ z に片側の天井をかける拘束 (Issue #137 Phase 3)。

# なぜ要るか

rotate_table_base の実機 run では、手先が教師と同等の水平方向の振幅・速度で
動いていたのに **降りる深さだけが足りず**、天板に触れずに空を切っていた。

| | 開始 z | ストローク最下点 z |
|---|---|---|
| 教師 1401 セグメント | median 200.0mm | **median 122.8mm** (p5 112.1 / p95 136.2) |
| 実機 Run1 (T0-1) 最初の 2.8s | 201.6mm | **147.9mm** |
| 実機 Run2 (L1+L4) 最初の 2.8s | 203.2mm | **192.4mm** |

横方向は外れていない (YOLO の hand_left × table_top の重なりが実測 97-100%)。
開始姿勢も教師 median とほぼ一致する。ズレているのは z だけ。

さらに時間方向のプロファイルを見ると、教師が開始直後に 200 → 132mm へ降りて
中盤 (t=30-80%) を 131-143mm で維持するのに対し、実機は **t=10% でいったん
上がり** (195 → 218mm)、中盤を高いまま通る。つまり「全体を一定量下げる」では
形が合わず、**中盤に上がってしまうのを押さえる片側拘束**が要る。

# 設計

- **絶対目標へのクランプ**であって、毎 tick の delta に定数を足すのではない。
  GR00T は `groot_relative_eef_v1` (state 相対) なので、delta に定数を足すと
  積算して単調に突っ込む (insert_and_tighten で R.elbow が 210/605 tick
  URDF 下限を割った件と同じ経路)。絶対目標に対する片側クランプならこれは
  原理的に起きない。
- **下げるだけ、持ち上げない**。天井より下にいる目標には触らない。
- **数値 Jacobian**。解析 (幾何) Jacobian は 0.195ms/tick で数値の 0.710ms より
  速いが、URDF chain の解釈を FK と二重に持つことになり、FK 側の変更で silent に
  ズレる。実測で両者の差は 2.1e-6 m/rad、コスト差は 33.3ms 予算の 1.5% しかない
  ので、FK の公開 API だけで完結する数値版を採る。
- **ハード下限**。教師 1401 セグメントの最下点 p5 = 112.1mm。力センサが無い以上
  「接触するまで降ろす」はできない (天板は軽い樹脂で滑るため反力もほとんど
  出ない) ので、幾何で止める。

# 精度 (教師 400 姿勢で実測、dz = -25mm 要求時)

| 減衰 λ | 誤差 median | p95 | 最大 | \\|Δq\\| 最大 |
|---|---|---|---|---|
| 0.01 | 0.14mm | 0.30mm | 0.51mm | 0.038 rad |
| 0.05 | 0.38mm | 0.59mm | 0.89mm | 0.038 rad |
| 0.1 | 1.16mm | 1.56mm | 2.00mm | 0.037 rad |

`‖J_z‖` の最小値は 400 姿勢で 0.379 m/rad = この作業領域に特異姿勢は無い。
既定 λ=0.01。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# VlaSkill 19D layout 内の左腕 7 joint と、SDK 29-joint 上の対応 index。
LEFT_ARM_19D = slice(3, 10)
LEFT_ARM_SDK29 = slice(15, 22)
WAIST_19D = slice(0, 3)
WAIST_SDK29 = slice(12, 15)
ARMS_19D = slice(3, 17)
ARMS_SDK29 = slice(15, 29)


@dataclass(frozen=True)
class ZCeilingResult:
    """1 tick 分のクランプ結果。

    Attributes:
        target_19d: クランプ後の 19D 絶対目標 (未発火なら入力と同一値)。
        bind: 天井に当たって補正したか。
        z_before_m / z_after_m: 補正前後の左手先 z [m]。
        correction_m: 実際に下げた量 [m] (負値)。
        max_joint_step_rad: この tick で動かした左腕関節の最大絶対値 [rad]。
        floor_rejected: DLS の解が `hard_floor_m` を割ったため補正を破棄したか。
            True のとき `target_19d` は入力そのままで `bind` は False。
    """

    target_19d: np.ndarray
    bind: bool
    z_before_m: float
    z_after_m: float
    correction_m: float
    max_joint_step_rad: float
    floor_rejected: bool = False


class LeftHandZCeiling:
    """19D 絶対目標の左手先 z を `ceiling_m` 以下に押し下げる。

    Args:
        fk: `G1WristFK`。`left_tool_position` を叩く。
        hard_floor_m: 天井として許す最小値 [m]。`apply` はこれを下回る
            `ceiling_m` を拒否する。教師 1401 セグメントの最下点 p5 = 0.1121 が
            既定の根拠。下げ幅そのものは天井までで頭打ちになるので、実行時の
            クランプは要らない (天井 >= 下限 が保証されているため)。
        max_correction_m: 1 tick で下げる最大量 [m]。突っ込み防止。
        damping: DLS の減衰 λ。
        max_joint_step_rad: 1 tick で動かす関節の絶対上限 [rad]。特異姿勢での
            発散に対する最後の蓋。
        fd_step_rad: 数値 Jacobian の前進差分ステップ [rad]。
    """

    def __init__(
        self,
        fk: Any,
        *,
        hard_floor_m: float,
        max_correction_m: float = 0.005,
        damping: float = 0.01,
        max_joint_step_rad: float = 0.05,
        fd_step_rad: float = 1e-5,
    ) -> None:
        if not hasattr(fk, "left_tool_position"):
            raise TypeError("fk must provide left_tool_position(joint_positions)")
        if not 0.0 < hard_floor_m < 1.0:
            raise ValueError(f"hard_floor_m must be in (0, 1) m, got {hard_floor_m}")
        for name, value in (
            ("max_correction_m", max_correction_m),
            ("damping", damping),
            ("max_joint_step_rad", max_joint_step_rad),
            ("fd_step_rad", fd_step_rad),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")
        self._fk = fk
        self._hard_floor_m = float(hard_floor_m)
        self._max_correction_m = float(max_correction_m)
        self._damping = float(damping)
        self._max_joint_step_rad = float(max_joint_step_rad)
        self._fd_step_rad = float(fd_step_rad)

    @property
    def hard_floor_m(self) -> float:
        return self._hard_floor_m

    def left_tool_z(self, target_19d: np.ndarray) -> float:
        """19D 絶対目標が実現する左手先 z [m]。"""
        return float(self._fk.left_tool_position(self._to_sdk29(target_19d))[2])

    def apply(
        self, target_19d: np.ndarray, ceiling_m: float | None
    ) -> ZCeilingResult:
        """天井を超えていれば左腕 7 関節で押し下げる。

        Args:
            target_19d: (19,) 絶対目標 [waist3, arms14, hand2]。
            ceiling_m: 天井 [m]。None なら何もしない (段 1 = 素のモデル)。

        Returns:
            ZCeilingResult。`target_19d` は入力を破壊しない新しい配列。
        """
        target = np.asarray(target_19d, dtype=np.float64).copy()
        if target.shape != (19,):
            raise ValueError(f"target_19d must be (19,), got {target.shape}")
        z_before = self.left_tool_z(target)
        if ceiling_m is None or z_before <= ceiling_m:
            return ZCeilingResult(
                target_19d=target,
                bind=False,
                z_before_m=z_before,
                z_after_m=z_before,
                correction_m=0.0,
                max_joint_step_rad=0.0,
            )
        if ceiling_m < self._hard_floor_m:
            raise ValueError(
                f"ceiling_m {ceiling_m} is below hard_floor_m {self._hard_floor_m}"
            )
        # 下げ幅は天井までの超過分。1 tick 上限で切る。天井は `hard_floor_m`
        # 以上であることを上で検証済なので、ここでの下限クランプは要らない。
        dz = max(ceiling_m - z_before, -self._max_correction_m)
        dq = self._solve_dq(target, dz)
        target[LEFT_ARM_19D] += dq
        z_after = self.left_tool_z(target)
        if z_after < self._hard_floor_m:
            # `_solve_dq` は DLS の近似解なので、入口の `ceiling_m >= hard_floor_m`
            # だけでは出力側の下限を保証できない。下回ったら補正を**破棄**する
            # (raise しないのは、実機 run 中の例外が controlled release に落ちて
            # かえって危ないため)。破棄後の目標は天井より必ず浅いので安全側。
            untouched = np.asarray(target_19d, dtype=np.float64).copy()
            return ZCeilingResult(
                target_19d=untouched,
                bind=False,
                z_before_m=z_before,
                z_after_m=z_before,
                correction_m=0.0,
                max_joint_step_rad=0.0,
                floor_rejected=True,
            )
        return ZCeilingResult(
            target_19d=target,
            bind=True,
            z_before_m=z_before,
            z_after_m=z_after,
            correction_m=z_after - z_before,
            max_joint_step_rad=float(np.max(np.abs(dq))) if dq.size else 0.0,
        )

    def lift_to(self, target_19d: np.ndarray, z_target_m: float) -> ZCeilingResult:
        """左手先を `z_target_m` まで **持ち上げる** (retry の戻り動作用)。

        `apply` と同じ Jacobian / 上限を使う逆向きの片側拘束。既に目標高さ以上に
        いる場合は何もしない。`hard_floor_m` は下げる方向の制約なのでここでは
        参照しない。

        Args:
            target_19d: (19,) 絶対目標。
            z_target_m: 持ち上げ先 [m]。

        Returns:
            ZCeilingResult。`bind` は持ち上げたかどうか。
        """
        target = np.asarray(target_19d, dtype=np.float64).copy()
        if target.shape != (19,):
            raise ValueError(f"target_19d must be (19,), got {target.shape}")
        z_before = self.left_tool_z(target)
        if z_before >= z_target_m:
            return ZCeilingResult(
                target_19d=target, bind=False, z_before_m=z_before,
                z_after_m=z_before, correction_m=0.0, max_joint_step_rad=0.0,
            )
        dz = min(z_target_m - z_before, self._max_correction_m)
        dq = self._solve_dq(target, dz)
        target[LEFT_ARM_19D] += dq
        z_after = self.left_tool_z(target)
        return ZCeilingResult(
            target_19d=target, bind=True, z_before_m=z_before, z_after_m=z_after,
            correction_m=z_after - z_before,
            max_joint_step_rad=float(np.max(np.abs(dq))) if dq.size else 0.0,
        )

    def _solve_dq(self, target_19d: np.ndarray, dz: float) -> np.ndarray:
        """z を `dz` 動かす左腕 7 関節の増分 (DLS + 関節上限)。"""
        jz = self._numeric_dz_jacobian(target_19d)
        dq = jz * (dz / (float(jz @ jz) + self._damping**2))
        step = float(np.max(np.abs(dq))) if dq.size else 0.0
        if step > self._max_joint_step_rad:
            dq = dq * (self._max_joint_step_rad / step)
        return dq

    def _numeric_dz_jacobian(self, target_19d: np.ndarray) -> np.ndarray:
        """左腕 7 関節に対する z の前進差分 Jacobian (7,) [m/rad]。"""
        base = self._to_sdk29(target_19d)
        z0 = float(self._fk.left_tool_position(base)[2])
        out = np.empty(7, dtype=np.float64)
        probe = base.copy()
        indices = range(LEFT_ARM_SDK29.start, LEFT_ARM_SDK29.stop)
        for k, j in enumerate(indices):
            probe[j] = base[j] + self._fd_step_rad
            out[k] = (float(self._fk.left_tool_position(probe)[2]) - z0) / self._fd_step_rad
            probe[j] = base[j]
        return out

    @staticmethod
    def _to_sdk29(target_19d: np.ndarray) -> np.ndarray:
        """19D 絶対目標 → FK 用 (29,)。leg [0:12] は左 chain に含まれないので 0。"""
        jp = np.zeros(29, dtype=np.float64)
        jp[WAIST_SDK29] = target_19d[WAIST_19D]
        jp[ARMS_SDK29] = target_19d[ARMS_19D]
        return jp
