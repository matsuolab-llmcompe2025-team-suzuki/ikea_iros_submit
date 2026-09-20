"""腕 kinematics の共通型 (Issue #123)。

EE pose の規約は **dataset (`action.ee_action`) と厳密に一致させる**:
    root_link (骨盤) 基準、`[x, y, z, roll, pitch, yaw]`、Euler XYZ (extrinsic)、
    位置 [m] / 角度 [rad]。
`model/subtask_policy_training/README.md` が明記する dataset 規約と同じ。
ここを揃えておくことで、dataset の数値をそのまま目標ポーズ YAML に持ち込める。

回転補間は rpy の線形補間ではなく **回転行列の測地線補間** を使う。
rpy 線形補間は pitch が ±pi/2 に近いと中間姿勢が破綻するため
(pick_table_leg の把持 pitch は約 0.83 rad = 47 deg で余裕はあるが、
目標ポーズを YAML で動かす前提なので安全側に倒す)。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# 片腕の関節数 (G1 は肩3 + 肘1 + 手首3)。
NUM_ARM_JOINTS_PER_SIDE: int = 7


class Side(str, Enum):
    """どちらの腕か。`str` 継承で YAML の "left"/"right" から直接引ける。"""

    LEFT = "left"
    RIGHT = "right"


class IKStatus(str, Enum):
    """IK の終了状態。

    caller (skill の stage) は `OK` 以外を **必ず abort 判定に使う**こと。
    黙って近似解を使うと、届いていないのに次 stage へ進んでしまう。
    """

    OK = "ok"
    NOT_CONVERGED = "not_converged"   # 反復上限まで回っても tolerance に入らない
    JOINT_LIMIT = "joint_limit"       # 解は出たが関節上限を超える
    SINGULAR = "singular"             # ヤコビアンが特異で解けない


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Euler XYZ (extrinsic) `[roll, pitch, yaw]` → 3x3 回転行列。

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """3x3 回転行列 → Euler XYZ (extrinsic) `[roll, pitch, yaw]`。

    `rpy_to_matrix` の逆。gimbal lock (|pitch| = pi/2) では roll を 0 に寄せる
    慣例的な分岐を取る。
    """
    R = np.asarray(R, dtype=np.float64)
    sp = -R[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = np.arcsin(sp)
    if abs(sp) > 1.0 - 1e-9:
        # gimbal lock: roll と yaw が縮退するので roll=0 に固定して yaw に寄せる。
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _rotation_geodesic(R_a: np.ndarray, R_b: np.ndarray, t: float) -> np.ndarray:
    """R_a から R_b への測地線補間 (t=0 で R_a、t=1 で R_b)。

    相対回転を axis-angle に落として t 倍し、Rodrigues で戻す。
    """
    R_rel = R_a.T @ R_b
    cos_theta = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return R_b.copy()
    # axis-angle 抽出 (theta が pi 近傍でも安定するよう対称成分から復元)。
    if abs(theta - np.pi) < 1e-6:
        # theta ~ pi: R_rel は対称。(R_rel + I)/2 の対角から軸を取る。
        M = (R_rel + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(M), 0.0, None))
        # 符号は非対角成分から決める (最大成分を基準に)。
        k = int(np.argmax(axis))
        if axis[k] > 1e-9:
            axis = M[:, k] / axis[k]
        norm = float(np.linalg.norm(axis))
        axis = axis / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array(
            [
                R_rel[2, 1] - R_rel[1, 2],
                R_rel[0, 2] - R_rel[2, 0],
                R_rel[1, 0] - R_rel[0, 1],
            ]
        ) / (2.0 * np.sin(theta))
    a = theta * float(t)
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    R_step = np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)
    return R_a @ R_step


@dataclass(frozen=True)
class EEPose:
    """root_link 基準の EE pose。`position` [m] (3,) / `rpy` [rad] (3,)。"""

    position: np.ndarray
    rpy: np.ndarray

    def __post_init__(self) -> None:
        for field_name in ("position", "rpy"):
            v = np.asarray(getattr(self, field_name), dtype=np.float64)
            if v.shape != (3,):
                raise ValueError(
                    f"EEPose.{field_name}: must have shape (3,), got {v.shape}"
                )
            if not np.all(np.isfinite(v)):
                raise ValueError(f"EEPose.{field_name}: contains NaN/Inf")
            object.__setattr__(self, field_name, v)

    @classmethod
    def from_vec6(cls, vec: object) -> "EEPose":
        """`[x, y, z, roll, pitch, yaw]` の 6 要素列から構築 (YAML / dataset 用)。"""
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if v.shape != (6,):
            raise ValueError(f"EEPose.from_vec6: expected 6 values, got {v.shape[0]}")
        return cls(position=v[:3], rpy=v[3:])

    def to_vec6(self) -> np.ndarray:
        return np.concatenate([self.position, self.rpy])

    @property
    def matrix(self) -> np.ndarray:
        """3x3 回転行列表現。"""
        return rpy_to_matrix(self.rpy)

    def position_error(self, other: "EEPose") -> float:
        """位置誤差のノルム [m]。"""
        return float(np.linalg.norm(self.position - other.position))

    def rotation_error(self, other: "EEPose") -> float:
        """姿勢誤差 [rad] (相対回転の回転角。rpy 各成分の差ではない)。"""
        R_rel = self.matrix.T @ other.matrix
        cos_theta = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
        return float(np.arccos(cos_theta))

    def interpolate(self, goal: "EEPose", t: float) -> "EEPose":
        """self → goal の補間 (位置は直線、姿勢は測地線)。`t` は [0, 1] にクリップ。

        stage の Cartesian 直線接近はこれを毎 tick 呼んで waypoint を作る。
        """
        s = float(np.clip(t, 0.0, 1.0))
        pos = self.position + (goal.position - self.position) * s
        R = _rotation_geodesic(self.matrix, goal.matrix, s)
        return EEPose(position=pos, rpy=matrix_to_rpy(R))


@dataclass(frozen=True)
class IKResult:
    """IK の結果。`status != OK` のとき `q` は使ってはならない。

    Attributes:
        status: 終了状態。`OK` 以外は stage が abort する。
        q: 関節角 (7,) [rad]。`status != OK` の場合は最終反復値 (診断用)。
        position_error: 収束時の位置誤差 [m]。
        rotation_error: 収束時の姿勢誤差 [rad]。
        iterations: 実際に回した反復回数。
    """

    status: IKStatus
    q: np.ndarray
    position_error: float
    rotation_error: float
    iterations: int

    @property
    def ok(self) -> bool:
        return self.status is IKStatus.OK

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=np.float64)
        if q.shape != (NUM_ARM_JOINTS_PER_SIDE,):
            raise ValueError(
                f"IKResult.q: must have shape ({NUM_ARM_JOINTS_PER_SIDE},), "
                f"got {q.shape}"
            )
        object.__setattr__(self, "q", q)
