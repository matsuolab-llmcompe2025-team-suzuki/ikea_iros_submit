"""G1 片腕 7-DoF の FK / IK 実装 (Issue #123)。

## 運動学パラメータの出所

`inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf`
から抽出した joint origin / axis / limit をそのまま持つ。URDF を parse せずに定数化して
あるのは、`urdf` は Orin 側資産で desktop の import 対象外 (CLAUDE.md) であり、
xml parse の依存を制御ループに持ち込みたくないため。URDF を更新したときは
`scripts/` の抽出手順で本表を再生成すること。

鎖: pelvis → waist_yaw(z) → waist_roll(x) → waist_pitch(y) → torso_link
         → shoulder_pitch(y) → shoulder_roll(x) → shoulder_yaw(z) → elbow(y)
         → wrist_roll(x) → wrist_pitch(y) → wrist_yaw(z) → dex1_base (fixed)

## URDF の妥当性検証

dataset (`observation.state.robot_q_current` と `observation.state.ee_state`) で
突き合わせた結果、回転表現は XYZ euler、軸符号は URDF どおりで整合した
(他の候補は残差 15-40 deg で明確に棄却)。残る残差 2-4 deg は **teleop 記録側の
時間ずれ**で、frame を -4 ずらすと 1.9 deg まで単調に減少する。つまり運動学ではなく
記録の同期の問題。

このため **目標ポーズは dataset の `ee_action` をそのまま使わず、本 module の FK を
dataset の関節角に適用して再導出する**。こうすると目標と IK が同じ定義になり、
記録遅延の影響を受けない。

## 手法

減衰最小二乗 (damped least squares)。テーブル前面は開空間で障害物が無いため
OMPL / cuRobo 級のプランナは使わない。Cartesian 直線補間 (skill 側) + 本 IK +
関節速度クリップで足りる。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from inference.desktop.lower_policy.kinematics.types import (
    NUM_ARM_JOINTS_PER_SIDE,
    EEPose,
    IKResult,
    IKStatus,
    Side,
    matrix_to_rpy,
    rpy_to_matrix,
)

# ---------------------------------------------------------------- URDF 定数

# (origin_xyz [m], origin_rpy [rad], axis) — axis は回転軸 (revolute のみ)
_WAIST: tuple[tuple[tuple[float, float, float], tuple[float, float, float], tuple[int, int, int]], ...] = (
    ((0.0, 0.0, 0.0),              (0.0, 0.0, 0.0), (0, 0, 1)),   # waist_yaw
    ((-0.0039635, 0.0, 0.044),     (0.0, 0.0, 0.0), (1, 0, 0)),   # waist_roll
    ((0.0, 0.0, 0.0),              (0.0, 0.0, 0.0), (0, 1, 0)),   # waist_pitch -> torso_link
)

_ARM: dict[Side, tuple] = {
    Side.LEFT: (
        ((0.0039563, 0.10022, 0.24778), (0.27931, 5.4949e-05, -0.00019159), (0, 1, 0)),
        ((0.0, 0.038, -0.013831),       (-0.27925, 0.0, 0.0),               (1, 0, 0)),
        ((0.0, 0.00624, -0.1032),       (0.0, 0.0, 0.0),                    (0, 0, 1)),
        ((0.015783, 0.0, -0.080518),    (0.0, 0.0, 0.0),                    (0, 1, 0)),
        ((0.100, 0.00188791, -0.010),   (0.0, 0.0, 0.0),                    (1, 0, 0)),
        ((0.038, 0.0, 0.0),             (0.0, 0.0, 0.0),                    (0, 1, 0)),
        ((0.051, 0.0, 0.0),             (0.0, 0.0, 0.0),                    (0, 0, 1)),
    ),
    Side.RIGHT: (
        ((0.0039563, -0.10021, 0.24778), (-0.27931, 5.4949e-05, 0.00019159), (0, 1, 0)),
        ((0.0, -0.038, -0.013831),       (0.27925, 0.0, 0.0),                (1, 0, 0)),
        ((0.0, -0.00624, -0.1032),       (0.0, 0.0, 0.0),                    (0, 0, 1)),
        ((0.015783, 0.0, -0.080518),     (0.0, 0.0, 0.0),                    (0, 1, 0)),
        ((0.100, -0.00188791, -0.010),   (0.0, 0.0, 0.0),                    (1, 0, 0)),
        ((0.038, 0.0, 0.0),              (0.0, 0.0, 0.0),                    (0, 1, 0)),
        ((0.051, 0.0, 0.0),              (0.0, 0.0, 0.0),                    (0, 0, 1)),
    ),
}

# wrist_yaw_link -> dex1_base_link (URDF の *_base_joint、fixed)
DEX1_BASE_OFFSET: tuple[float, float, float] = (0.0415, 0.0, 0.0)

# URDF の可動域 [rad]。順序は肩pitch/肩roll/肩yaw/肘/手首roll/手首pitch/手首yaw。
_URDF_LIMITS: dict[Side, tuple[tuple[float, float], ...]] = {
    Side.LEFT: (
        (-3.0892, 2.6704), (-1.5882, 2.2515), (-2.618, 2.618), (-1.0472, 2.0944),
        (-1.972222054, 1.972222054), (-1.614429558, 1.614429558), (-1.614429558, 1.614429558),
    ),
    Side.RIGHT: (
        (-3.0892, 2.6704), (-2.2515, 1.5882), (-2.618, 2.618), (-1.0472, 2.0944),
        (-1.972222054, 1.972222054), (-1.614429558, 1.614429558), (-1.614429558, 1.614429558),
    ),
}

# G1ArmActuator の安全クランプ (pose_utils.POSE_ABS_LIMIT_RAD と同値)。
# URDF の機械限界より狭いので、IK はこちらを守る解を探す。
# dataset 実測では腕 14 関節が ±1.5 を超える frame は 0.44% しかなく、
# この範囲で task は成立する。
OPERATIONAL_LIMIT_RAD: float = 1.5


def _rot_axis(axis: Sequence[int], theta: float) -> np.ndarray:
    """軸まわり theta の回転行列 (Rodrigues)。"""
    a = np.asarray(axis, dtype=np.float64)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _log_so3(R: np.ndarray) -> np.ndarray:
    """回転行列 → 回転ベクトル (axis * angle)。IK の姿勢誤差に使う。"""
    cos_t = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if theta < 1e-9:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        # theta ~ pi: 対称成分から軸を復元する
        M = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(M), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 1e-9:
            axis = M[:, k] / axis[k]
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        return axis * theta
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (theta / (2.0 * np.sin(theta)))


class G1ArmKinematics:
    """`ArmKinematics` の実装。root_link (骨盤) 基準、固定ベース。

    Args:
        tcp_offset: EE 原点を dex1_base_link からさらにずらす量 [m] (3,)。
            グリッパ指先を基準にしたい場合に使う。default は dex1_base_link そのもの。
        limit_rad: 関節の運用上限 [rad]。URDF 限界と AND を取る。
        damping: 減衰係数 lambda。大きいほど特異点近傍で安定するが収束が鈍る。
        max_iterations / pos_tol / rot_tol: 反復の打ち切り条件。

            **収束閾値を task の許容より過度に厳しくしないこと。** stage の到達判定は
            位置 20mm / 姿勢 0.12 rad なので、IK 側を 0.1mm まで詰めると、実用上
            十分な解 (残差 0.3mm) に達していても反復上限で `NOT_CONVERGED` を返し、
            skill が誤って abort する。default はそれぞれ stage 許容の 1/20 程度に取り、
            「到達不能」だけを失敗として拾えるようにしてある。
        max_step_rad: 1 反復あたりの関節変化量の上限 (発散防止)。
    """

    def __init__(
        self,
        *,
        tcp_offset: Sequence[float] = (0.0, 0.0, 0.0),
        limit_rad: float = OPERATIONAL_LIMIT_RAD,
        damping: float = 0.05,
        max_iterations: int = 80,
        pos_tol: float = 1e-3,
        rot_tol: float = 5e-3,
        max_step_rad: float = 0.2,
    ) -> None:
        tcp = np.asarray(tcp_offset, dtype=np.float64).reshape(-1)
        if tcp.shape != (3,):
            raise ValueError(f"tcp_offset: must have shape (3,), got {tcp.shape}")
        self._tcp = np.asarray(DEX1_BASE_OFFSET, dtype=np.float64) + tcp
        self._damping = float(damping)
        self._max_iter = int(max_iterations)
        self._pos_tol = float(pos_tol)
        self._rot_tol = float(rot_tol)
        self._max_step = float(max_step_rad)
        lim = abs(float(limit_rad))
        self._limits: dict[Side, np.ndarray] = {
            side: np.array(
                [(max(lo, -lim), min(hi, lim)) for lo, hi in _URDF_LIMITS[side]],
                dtype=np.float64,
            )
            for side in (Side.LEFT, Side.RIGHT)
        }

    # ------------------------------------------------------------ FK

    def joint_limits(self, side: Side) -> np.ndarray:
        """(7, 2) の [lower, upper]。URDF 限界と運用上限の AND。"""
        return self._limits[side].copy()

    def _frames(
        self, q: np.ndarray, side: Side, waist: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """EE の位置 / 回転と、各腕関節の原点 / 回転軸 (world) を返す。

        後半 2 つは幾何ヤコビアンの構成に使う。
        """
        p = np.zeros(3)
        R = np.eye(3)
        for (xyz, rpy, axis), th in zip(_WAIST, waist):
            p = p + R @ np.asarray(xyz, dtype=np.float64)
            R = R @ rpy_to_matrix(np.asarray(rpy)) @ _rot_axis(axis, float(th))

        origins: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        for (xyz, rpy, axis), th in zip(_ARM[side], q):
            p = p + R @ np.asarray(xyz, dtype=np.float64)
            R = R @ rpy_to_matrix(np.asarray(rpy))
            origins.append(p.copy())
            axes.append(R @ np.asarray(axis, dtype=np.float64))
            R = R @ _rot_axis(axis, float(th))
        p = p + R @ self._tcp
        return p, R, origins, axes

    @staticmethod
    def _check(q: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if arr.shape != (NUM_ARM_JOINTS_PER_SIDE,):
            raise ValueError(
                f"{name}: must have shape ({NUM_ARM_JOINTS_PER_SIDE},), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: contains NaN/Inf")
        return arr

    @staticmethod
    def _waist(waist: Optional[np.ndarray]) -> np.ndarray:
        if waist is None:
            return np.zeros(3)
        arr = np.asarray(waist, dtype=np.float64).reshape(-1)
        if arr.shape != (3,):
            raise ValueError(f"waist: must have shape (3,), got {arr.shape}")
        return arr

    def fk(
        self, q: np.ndarray, side: Side, waist: Optional[np.ndarray] = None
    ) -> EEPose:
        p, R, _, _ = self._frames(
            self._check(q, "fk: q"), side, self._waist(waist)
        )
        return EEPose(position=p, rpy=matrix_to_rpy(R))

    def jacobian(
        self, q: np.ndarray, side: Side, waist: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """(6, 7) の幾何ヤコビアン。上 3 行が並進、下 3 行が回転。"""
        p, _, origins, axes = self._frames(
            self._check(q, "jacobian: q"), side, self._waist(waist)
        )
        J = np.zeros((6, NUM_ARM_JOINTS_PER_SIDE))
        for i in range(NUM_ARM_JOINTS_PER_SIDE):
            J[0:3, i] = np.cross(axes[i], p - origins[i])
            J[3:6, i] = axes[i]
        return J

    # ------------------------------------------------------------ IK

    def ik(
        self,
        target: EEPose,
        seed: np.ndarray,
        side: Side,
        waist: Optional[np.ndarray] = None,
    ) -> IKResult:
        """減衰最小二乗による反復解法。

        失敗は例外ではなく `IKStatus` で返す (制御ループを壊さないため)。

        減衰最小二乗は seed 依存の局所解に嵌ることがあり、到達可能な目標でも
        1 回の反復では解けないことがある (実測: 冷 seed の格子探索で、到達可能な
        領域の内側に飛び飛びの失敗が出る)。そこで失敗したら **決定論的な予備 seed** を
        順に試す。乱数を使わないので同じ入力なら常に同じ解になる。
        予備 seed は失敗時にしか走らないので、通常時の計算量は変わらない。
        """
        primary = self._check(seed, "ik: seed")
        result = self._solve(target, primary, side, waist)
        if result.ok:
            return result
        best = result
        for candidate in self._fallback_seeds(side, primary):
            attempt = self._solve(target, candidate, side, waist)
            if attempt.ok:
                return attempt
            if attempt.position_error < best.position_error:
                best = attempt
        return best

    def _fallback_seeds(self, side: Side, primary: np.ndarray) -> list[np.ndarray]:
        """局所解から抜けるための予備 seed (決定論)。

        可動域の中央、ゼロ姿勢、肘を曲げた 2 通り。腕の主要な分岐 (肘の向き) を
        跨ぐように選んである。
        """
        lim = self._limits[side]
        mid = (lim[:, 0] + lim[:, 1]) / 2.0
        elbow_sign = 1.0 if side is Side.LEFT else 1.0
        bent = mid.copy()
        bent[3] = np.clip(elbow_sign * 0.9, lim[3, 0], lim[3, 1])   # 肘
        folded = mid.copy()
        folded[0] = np.clip(-0.8, lim[0, 0], lim[0, 1])             # 肩 pitch
        folded[3] = np.clip(elbow_sign * 1.4, lim[3, 0], lim[3, 1])
        out = [mid, np.zeros(NUM_ARM_JOINTS_PER_SIDE), bent, folded]
        return [
            np.clip(s, lim[:, 0], lim[:, 1])
            for s in out
            if not np.allclose(s, primary, atol=1e-9)
        ]

    def _solve(
        self,
        target: EEPose,
        seed: np.ndarray,
        side: Side,
        waist: Optional[np.ndarray],
    ) -> IKResult:
        q = np.clip(
            np.asarray(seed, dtype=np.float64),
            self._limits[side][:, 0],
            self._limits[side][:, 1],
        )
        w = self._waist(waist)
        R_target = target.matrix
        lam2 = self._damping ** 2
        pos_err = rot_err = float("inf")

        for it in range(1, self._max_iter + 1):
            p, R, origins, axes = self._frames(q, side, w)
            e_pos = target.position - p
            e_rot = _log_so3(R_target @ R.T)
            pos_err = float(np.linalg.norm(e_pos))
            rot_err = float(np.linalg.norm(e_rot))
            if pos_err <= self._pos_tol and rot_err <= self._rot_tol:
                return IKResult(IKStatus.OK, q, pos_err, rot_err, it)

            J = np.zeros((6, NUM_ARM_JOINTS_PER_SIDE))
            for i in range(NUM_ARM_JOINTS_PER_SIDE):
                J[0:3, i] = np.cross(axes[i], p - origins[i])
                J[3:6, i] = axes[i]

            # 特異点そのものは減衰で扱えるが、完全に階数落ちした場合は解けない。
            if float(np.linalg.matrix_rank(J, tol=1e-8)) < 1:
                return IKResult(IKStatus.SINGULAR, q, pos_err, rot_err, it)

            e = np.concatenate([e_pos, e_rot])
            try:
                dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            except np.linalg.LinAlgError:
                return IKResult(IKStatus.SINGULAR, q, pos_err, rot_err, it)

            step = float(np.abs(dq).max())
            if step > self._max_step:
                dq = dq * (self._max_step / step)
            q = np.clip(q + dq, self._limits[side][:, 0], self._limits[side][:, 1])

        # 反復を使い切った。関節上限に貼り付いているなら原因はそちらと報告する。
        at_limit = np.isclose(q, self._limits[side][:, 0]) | np.isclose(
            q, self._limits[side][:, 1]
        )
        status = IKStatus.JOINT_LIMIT if at_limit.any() else IKStatus.NOT_CONVERGED
        return IKResult(status, q, pos_err, rot_err, self._max_iter)
