"""boundary ↔ desktop orchestrator の I/O アダプタ層 (full orchestrator 再利用の土台)。

原さんの orchestrator (issue/128) を boundary Policy から駆動するための glue。
orchestrator は DDS I/O (JointStateSource / Dex1StateSource / arm・waist・hand actuator /
Ros2FrameSource) に密結合しているので、それらを boundary obs/action に差し替える。

- `BoundaryJointStateSource`: boundary body_q(29) を注入し get() で JointStateData 互換を返す。
- `BoundaryDex1StateSource`: Dex1 開度を注入し get() で Dex1StateData 互換を返す。
- `InterceptorActuator`: send_action(a) で最新 action を捕捉 (robot へ送らない)。
  VlaSkill は 19D を waist3→waist_actuator / arms14→dispatcher return / hand2→hand_actuator に
  分配するので、waist・hand を interceptor 化し、arms は tick 結果 (TickResult.action) から取る。
- `assemble_19d`: 捕捉した waist3 + arms14 + hand2 → 19D。
- `build_frame_data`: boundary head RGB → FrameData(rgb, t) (BGR、packed stereo は複製で近似)。

いずれも duck-typed (get()/send_action) なので、実 orchestrator は
JointStateSourceProtocol / WaistActuator / HandActuator としてこれらを受け取れる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# G1 29-DoF joint 名 (inference.desktop ... joint_mapping.G1_JOINT_NAMES と同順)。
G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

DEX1_OPEN_VALUE = 4.5   # Dex1 physical open [rad]


@dataclass
class _JointStateData:
    """desktop JointStateData 互換 (name/position/velocity/effort/t)。"""
    name: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    t: int


@dataclass
class _Dex1StateData:
    """desktop Dex1StateData 互換 (position_rad/…/t)。"""
    position_rad: np.ndarray
    left_received_monotonic_ns: int
    right_received_monotonic_ns: int
    t: int


class BoundaryJointStateSource:
    """boundary body_q(29) を注入 → orchestrator が get() で読む。"""

    def __init__(self) -> None:
        self._latest: _JointStateData | None = None

    def update(self, body_q29: np.ndarray, t: int) -> None:
        q = np.asarray(body_q29, dtype=np.float64)
        if q.shape != (29,):
            raise ValueError(f"body_q must be (29,), got {q.shape}")
        z = np.zeros(29, dtype=np.float64)
        self._latest = _JointStateData(
            name=G1_JOINT_NAMES, position=q, velocity=z, effort=z, t=int(t)
        )

    def get(self) -> _JointStateData | None:
        return self._latest


class BoundaryDex1StateSource:
    """Dex1 開度 (fraction 2) を注入 → get() で Dex1StateData 互換。"""

    def __init__(self, open_fraction: tuple[float, float] = (1.0, 1.0)) -> None:
        self._frac = np.clip(np.asarray(open_fraction, np.float64), 0.0, 1.0)
        self._t = 0

    def update(self, open_fraction, t: int) -> None:
        self._frac = np.clip(np.asarray(open_fraction, np.float64), 0.0, 1.0)
        self._t = int(t)

    def get(self) -> _Dex1StateData:
        return _Dex1StateData(
            position_rad=(self._frac * DEX1_OPEN_VALUE).astype(np.float64),
            left_received_monotonic_ns=self._t,
            right_received_monotonic_ns=self._t,
            t=self._t,
        )


class InterceptorActuator:
    """send_action(a) を捕捉 (robot へ送らない)。waist/hand actuator の差し替え用。"""

    def __init__(self, name: str = "interceptor") -> None:
        self.name = name
        self.last: np.ndarray | None = None
        self.count = 0

    def send_action(self, action: np.ndarray) -> None:
        self.last = np.asarray(action, dtype=np.float64).reshape(-1)
        self.count += 1

    # 実 actuator が持ちうる lifecycle no-op (呼ばれても安全)。
    def start(self, *a: Any, **k: Any) -> None:  # noqa: D401
        return None

    def stop(self, *a: Any, **k: Any) -> None:
        return None

    def reset(self) -> None:
        self.last = None


@dataclass
class BoundaryFrameData:
    """desktop FrameData 互換 (rgb: HWC BGR, t)。"""
    rgb: np.ndarray
    t: int


class BoundaryWristSource:
    """boundary wrist 画像 (BGR) を注入 → orchestrator が get() で FrameData を読む。"""

    def __init__(self) -> None:
        self._latest: BoundaryFrameData | None = None

    def update(self, bgr: np.ndarray, t: int) -> None:
        self._latest = BoundaryFrameData(
            rgb=np.ascontiguousarray(np.asarray(bgr, dtype=np.uint8)), t=int(t)
        )

    def get(self) -> "BoundaryFrameData | None":
        return self._latest


def build_frame_data(head_bgr: np.ndarray, t: int, packed_stereo: bool = True) -> BoundaryFrameData:
    """boundary head 画像 (BGR) → FrameData。

    orchestrator は head を packed stereo (左右連結) と想定し、perception には左眼、
    policy には左右を渡す。boundary は単一 head なので、packed_stereo=True の時は
    横に複製して packed 幅にする (head_perception_view='left' で左半分=元画像が使われる)。
    """
    img = np.ascontiguousarray(np.asarray(head_bgr, dtype=np.uint8))
    if packed_stereo:
        img = np.concatenate([img, img], axis=1)   # (H, 2W, 3)
    return BoundaryFrameData(rgb=img, t=int(t))


def assemble_19d(
    waist3: np.ndarray | None, arms14: np.ndarray, hand2: np.ndarray | None
) -> np.ndarray:
    """捕捉した各 actuator target → 19D (waist3 + arms14 + hand2)。

    waist/hand が未捕捉 (dispatch されない skill) の場合は 0 埋め。
    """
    out = np.zeros(19, dtype=np.float64)
    if waist3 is not None:
        out[0:3] = np.asarray(waist3, np.float64).reshape(-1)[:3]
    out[3:17] = np.asarray(arms14, np.float64).reshape(-1)[:14]
    if hand2 is not None:
        out[17:19] = np.asarray(hand2, np.float64).reshape(-1)[:2]
    return out
