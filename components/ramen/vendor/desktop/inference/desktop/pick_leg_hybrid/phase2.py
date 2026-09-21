"""区間 2 の Motion planning (Issue #136、determinded.md D-10)。

区間 2 は「右手で脚を左手の位置まで運ぶ」。**可変始点 → 固定終点**:

- 始点: 1→2 に遷移した瞬間の EE pose (試行ごとに変わる。実行時に FK で得る)
- 終点: データセットから決めた区間 3 との境界姿勢 (固定)

# IK が要らない理由

提出は boundary の `decoupled` lane で、腕は **手先の位置と姿勢**を送る
(`(T,25)` の `[4:18]`)。関節角への変換は運営側の IK が行う。
こちらが持つのは FK だけで、それは `perception/g1_urdf_fk.G1WristFK` にある。

そのため区間 2 は「現在の手先姿勢から目標の手先姿勢まで補間する」で足りる。
テーブル前面は開空間で、この区間で避けるべき障害物は無い。

# 未確定: EE の基準 frame

運営の IK が期待する EE 基準 frame は**未確定** (`taskspace_adapter` の調査 Q1)。
実測では `G1WristFK` の出力とデータセット記録の `ee_state` は
**左 99 mm / 右 129 mm** ずれる。`ee_frame_transform` で後から吸収する前提。
目標値をどちらの定義で置いたかは config に明記すること (D-3)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from inference.desktop.lower_policy.policies.taskspace_adapter import (
    DEFAULT_BASE_HEIGHT_M,
    TASKSPACE_DIM,
    dex1_model_to_taskspace,
)

#: `(T, 25)` の列割り (taskspace_adapter と同一、並べ替え禁止)。
SLICE_HAND_LEFT = slice(0, 2)
SLICE_HAND_RIGHT = slice(2, 4)
SLICE_EE_LEFT_POS = slice(4, 7)
SLICE_EE_LEFT_QUAT = slice(7, 11)
SLICE_EE_RIGHT_POS = slice(11, 14)
SLICE_EE_RIGHT_QUAT = slice(14, 18)
SLICE_NAVIGATE = slice(18, 21)
IDX_BASE_HEIGHT = 21
SLICE_TORSO_RPY = slice(22, 25)


@dataclass(frozen=True)
class EePose:
    """手先の位置と姿勢。quat は (w, x, y, z)。"""

    pos: np.ndarray
    quat: np.ndarray

    @staticmethod
    def of(pos: Sequence[float], quat: Sequence[float]) -> "EePose":
        p = np.asarray(pos, dtype=np.float64).reshape(-1)
        q = np.asarray(quat, dtype=np.float64).reshape(-1)
        if p.shape != (3,):
            raise ValueError(f"pos must be (3,), got {p.shape}")
        if q.shape != (4,):
            raise ValueError(f"quat must be (4,) wxyz, got {q.shape}")
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n < 1e-9:
            raise ValueError(f"quat must be a non-zero finite quaternion, got {q}")
        if not np.all(np.isfinite(p)):
            raise ValueError(f"pos must be finite, got {p}")
        return EePose(pos=p, quat=q / n)


@dataclass(frozen=True)
class Phase2Config:
    """区間 2 の設定。

    Attributes:
        goal_left / goal_right: 区間 3 との境界での手先姿勢 (固定終点)。
        duration_sec: 始点から終点まで何秒かけるか。
        pos_tol / rot_tol: 到達判定の許容 (位置 [m] / 姿勢 [rad])。
        hold_hand_value: 区間 2 の間、Dex1 に出し続ける model 空間の開度。
            脚を保持し続けるため閉じたまま。
        ee_frame: 目標値がどの FK 定義で書かれているか。**必ず明記する** (D-3)。
    """

    goal_left: EePose
    goal_right: EePose
    duration_sec: float = 2.7
    pos_tol: float = 0.02
    rot_tol: float = 0.12
    hold_hand_value: float = 2.2
    hold_hand_left_value: Optional[float] = None
    hold_hand_right_value: Optional[float] = None
    ee_frame: str = "g1_wrist_fk"

    def __post_init__(self) -> None:
        if self.duration_sec <= 0.0:
            raise ValueError(f"duration_sec must be > 0, got {self.duration_sec}")
        if self.pos_tol <= 0.0 or self.rot_tol <= 0.0:
            raise ValueError("pos_tol and rot_tol must be > 0")

    @property
    def hold_left(self) -> float:
        return float(
            self.hold_hand_value
            if self.hold_hand_left_value is None
            else self.hold_hand_left_value
        )

    @property
    def hold_right(self) -> float:
        return float(
            self.hold_hand_value
            if self.hold_hand_right_value is None
            else self.hold_hand_right_value
        )


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """単位 quaternion (wxyz) の球面線形補間。

    Args:
        q0 / q1: (4,) 単位 quaternion (w, x, y, z)。
        s: 0..1。

    Returns:
        (4,) 単位 quaternion。
    """
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:  # 近い方の回転を選ぶ
        b = -b
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:  # ほぼ同じ → 線形で十分
        out = a + s * (b - a)
        return out / np.linalg.norm(out)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - s) * theta) / sin_theta
    w1 = np.sin(s * theta) / sin_theta
    out = w0 * a + w1 * b
    return out / np.linalg.norm(out)


def quat_angle_between(q0: np.ndarray, q1: np.ndarray) -> float:
    """2 つの quaternion の間の回転角 [rad] (0..pi)。"""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    dot = abs(float(np.dot(a, b)))
    return float(2.0 * np.arccos(min(1.0, max(-1.0, dot))))


def taskspace_row(
    left: EePose,
    right: EePose,
    *,
    hand_left_model: float,
    hand_right_model: float,
    navigate_cmd: Sequence[float] = (0.0, 0.0, 0.0),
    base_height_cmd: float = DEFAULT_BASE_HEIGHT_M,
    torso_rpy_cmd: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """手先姿勢と Dex1 開度から boundary の `(25,)` 1 行を作る。

    列割りは `taskspace_adapter` と同一。下半身は pick 系なので既定 0-hold。
    ただし `base_height_cmd` は 0 が「0 m へ沈め」の意味になるので既定は
    中立の `DEFAULT_BASE_HEIGHT_M` (= 0.74 m)。

    Args:
        left / right: 手先姿勢。
        hand_left_model / hand_right_model: model 空間の Dex1 開度
            (0=閉 .. 4.5=全開)。boundary の -1/+1 へは内部で変換する。

    Returns:
        (25,) float32。
    """
    out = np.zeros(TASKSPACE_DIM, dtype=np.float32)
    out[SLICE_HAND_LEFT] = dex1_model_to_taskspace(float(hand_left_model))
    out[SLICE_HAND_RIGHT] = dex1_model_to_taskspace(float(hand_right_model))
    out[SLICE_EE_LEFT_POS] = left.pos
    out[SLICE_EE_LEFT_QUAT] = left.quat
    out[SLICE_EE_RIGHT_POS] = right.pos
    out[SLICE_EE_RIGHT_QUAT] = right.quat
    out[SLICE_NAVIGATE] = np.asarray(navigate_cmd, dtype=np.float32)
    out[IDX_BASE_HEIGHT] = float(base_height_cmd)
    out[SLICE_TORSO_RPY] = np.asarray(torso_rpy_cmd, dtype=np.float32)
    return out


class Phase2Motion:
    """可変始点 → 固定終点の直線補間。

    `start()` で始点を捉え、`step(t)` で現在時刻ぶんの `(25,)` を返す。
    `reached()` で終点に着いたかを判定する (区間 2→3 の遷移条件)。
    """

    def __init__(self, cfg: Phase2Config) -> None:
        self.cfg = cfg
        self._t0: Optional[float] = None
        self._start_left: Optional[EePose] = None
        self._start_right: Optional[EePose] = None

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def start(self, t: float, left: EePose, right: EePose) -> None:
        """始点を捉える。1→2 に遷移した瞬間に 1 回だけ呼ぶ。"""
        self._t0 = float(t)
        self._start_left = left
        self._start_right = right

    def reset(self) -> None:
        self._t0 = None
        self._start_left = None
        self._start_right = None

    def progress(self, t: float) -> float:
        """0..1 の進捗。未開始なら 0。"""
        if self._t0 is None:
            return 0.0
        s = (float(t) - self._t0) / self.cfg.duration_sec
        return float(min(1.0, max(0.0, s)))

    def pose_at(self, t: float) -> tuple[EePose, EePose]:
        """時刻 t での左右の目標手先姿勢。"""
        if self._t0 is None or self._start_left is None or self._start_right is None:
            raise RuntimeError("Phase2Motion.start() must be called first")
        s = self.progress(t)
        return (
            _lerp_pose(self._start_left, self.cfg.goal_left, s),
            _lerp_pose(self._start_right, self.cfg.goal_right, s),
        )

    def step(self, t: float) -> np.ndarray:
        """時刻 t での boundary `(25,)` 指令。脚は掴んだまま保持する。"""
        left, right = self.pose_at(t)
        return taskspace_row(
            left,
            right,
            hand_left_model=self.cfg.hold_left,
            hand_right_model=self.cfg.hold_right,
        )

    def reached(self, left: EePose, right: EePose) -> bool:
        """**実測**の手先姿勢が終点の許容内に入ったか (区間 2→3 の条件)。

        補間の進捗ではなく実測で判定する。指令どおり動いていない場合に
        誤って次へ進まないため。
        """
        for actual, goal in ((left, self.cfg.goal_left), (right, self.cfg.goal_right)):
            if float(np.linalg.norm(actual.pos - goal.pos)) > self.cfg.pos_tol:
                return False
            if quat_angle_between(actual.quat, goal.quat) > self.cfg.rot_tol:
                return False
        return True


def _lerp_pose(a: EePose, b: EePose, s: float) -> EePose:
    return EePose(
        pos=a.pos + s * (b.pos - a.pos),
        quat=slerp(a.quat, b.quat, s),
    )
