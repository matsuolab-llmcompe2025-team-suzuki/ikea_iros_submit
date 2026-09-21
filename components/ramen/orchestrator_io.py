"""boundary ↔ desktop orchestrator の I/O アダプタ層 (full orchestrator 再利用の土台)。

原さんの orchestrator (issue/128) を boundary Policy から駆動するための glue。
orchestrator は DDS I/O (JointStateSource / Dex1StateSource / arm・waist・hand actuator /
Ros2FrameSource) に密結合しているので、それらを boundary obs/action に差し替える。

- `BoundaryJointStateSource`: boundary body_q(29) を注入し get() で JointStateData 互換を返す。
- `BoundaryDex1StateSource`: Dex1 開度を注入し get() で Dex1StateData 互換を返す。
- `MeasuredDex1StateSource`: 運営 `:5557` の `gripper_q` (実測) を注入。来ない間は
  fallback (合成) に委譲する。
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
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

DEX1_OPEN_VALUE = 4.5  # Dex1 physical open [rad]

# 運営リグの Dex1-1 生モータ角。`tools/run_wbc_with_dex1.py:49-50,65-66`:
#     q =  0.00  CLOSED
#     q = -5.30  OPEN     <- sign is FLIPPED vs Unitree's reference
# 我々の model 空間 (0=閉 .. DEX1_OPEN_VALUE=全開) とは **全開の表し方が違うだけ**で、
# 比率で写せば端点も中間も一致する (両方とも「その手の全開」を指している)。
DEX1_ORGANIZER_CLOSED_Q = 0.0
DEX1_ORGANIZER_OPEN_Q = -5.30

# 実測が途切れたとみなすまでの時間。カメラと同じ基準 (_CAMERA_STALE_TIMEOUT_S)。
# bridge は 50 Hz なので、これを超えるのは bridge 自体が止まったとき。
DEX1_MEASURED_STALE_S = 0.5


def gripper_q_to_model_rad(q: float) -> float:
    """運営の生モータ角 → こちらの model 空間 [0(閉) .. DEX1_OPEN_VALUE(全開)]。

    `taskspace_adapter.dex1_model_to_taskspace` →
    `run_wbc_with_dex1.hand_norm_to_dex1_q` の逆写像。往復すると元に戻る:

        model 4.5 → norm -1 → q -5.30 → model 4.5   (全開)
        model 0.0 → norm +1 → q  0.00 → model 0.0   (全閉)
    """
    span = DEX1_ORGANIZER_OPEN_Q - DEX1_ORGANIZER_CLOSED_Q
    frac_open = (float(q) - DEX1_ORGANIZER_CLOSED_Q) / span
    return float(np.clip(frac_open, 0.0, 1.0) * DEX1_OPEN_VALUE)


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


class MeasuredDex1StateSource:
    """`:5557` の `gripper_q` (実測) を注入 → get() で Dex1StateData 互換を返す。

    実測がまだ来ていない / 途切れた間は `fallback` に委譲する。`fallback` は
    `SyntheticDex1StateSource` (= 自分の hand 指令のエコー) を想定していて、
    **会場で bridge が古い版だったときも今までどおり動く**ようにしてある。

    実測が入ると何が変わるか:
      - `insert_table_leg` / `rotate_leg_to_tighten` の `hand_state` が
        「指令どおり閉じた仮定」ではなく実際の開度になる
      - `pick_leg_hybrid` の interlock (`実測 - 指令`) が意味を持つ。
        合成では差が恒等的に 0 なので一度も発火しない

    Args:
        fallback: 実測が無い間に使う source。`get()` を持てば何でもよい。
        stale_after_s: 最後の実測からこの時間を超えたら fallback に戻る。
    """

    def __init__(
        self, fallback: Any = None, *, stale_after_s: float = DEX1_MEASURED_STALE_S
    ) -> None:
        self._fallback = fallback
        self._stale_after_s = float(stale_after_s)
        self._latest: _Dex1StateData | None = None
        self._latest_obs_t: float | None = None
        self._obs_t: float | None = None
        #: 一度でも実測が取れたか。起動診断とログに使う (途切れても False に戻さない)。
        self.ever_measured = False

    def update(self, gripper_q: Any, t: int, *, obs_t: float | None = None) -> bool:
        """`obs["gripper_q"]` を取り込む。取り込めたら True。

        Args:
            gripper_q: `gripper_state.GripperStateStream.poll()` が返す生の dict、
                または None (bridge が載せていない / 旧 client)。
            t: 世代カウンタ (Dex1StateData.t に入れる)。
            obs_t: 運営 client の frame 時刻。鮮度判定に使う。None なら
                「常に新鮮」とみなす (時刻が取れない経路での退行を避ける)。
        """
        self._obs_t = obs_t
        if not isinstance(gripper_q, dict):
            return False
        try:
            positions = np.asarray(
                [
                    gripper_q_to_model_rad(gripper_q[side]["q"])
                    for side in ("left", "right")
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            return False
        if not np.all(np.isfinite(positions)):
            return False
        self._latest = _Dex1StateData(
            position_rad=positions,
            left_received_monotonic_ns=int(t),
            right_received_monotonic_ns=int(t),
            t=int(t),
        )
        self._latest_obs_t = obs_t
        self.ever_measured = True
        return True

    @property
    def measured_is_fresh(self) -> bool:
        """いま返すのが実測かどうか。"""
        if self._latest is None:
            return False
        if self._latest_obs_t is None or self._obs_t is None:
            return True
        return (self._obs_t - self._latest_obs_t) <= self._stale_after_s

    def get(self) -> Any:
        if self.measured_is_fresh:
            return self._latest
        return self._fallback.get() if self._fallback is not None else None


class InterceptorActuator:
    """send_action(a) を捕捉 (robot へ送らない)。waist/hand actuator の差し替え用。"""

    def __init__(self, name: str = "interceptor") -> None:
        self.name = name
        self.last: np.ndarray | None = None
        self.count = 0

    @property
    def latest(self) -> np.ndarray | None:
        """自前経路の hand actuator と同じ名前。

        `SyntheticDex1StateSource` が `getattr(source, "latest", None)` で直近の
        hand 指令を読む (`perception/dex1_state_source.py`)。`.last` のままだと
        **常に None に見えて frame-0 の seed 値を返し続ける**ので、名前を揃える。
        """
        return self.last

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
    """desktop FrameData 互換 (rgb: HWC BGR, t, received_monotonic_ns)。

    `received_monotonic_ns` は自前経路の `FrameData` と同じ意味で、**この frame が
    届いた時刻**。`assembly.build_observation` がここを読んで `camera_stale_roles`
    に渡す。持っていないと `or obs["t"]` の fallback で「常に今」になり、
    **止まったカメラを検出できない**。
    """

    rgb: np.ndarray
    t: int
    received_monotonic_ns: int = 0


class BoundaryWristSource:
    """boundary wrist 画像 (BGR) を注入 → orchestrator が get() で FrameData を読む。"""

    def __init__(self) -> None:
        self._latest: BoundaryFrameData | None = None

    def update(self, bgr: np.ndarray, t: int, received_monotonic_ns: int = 0) -> None:
        self._latest = BoundaryFrameData(
            rgb=np.ascontiguousarray(np.asarray(bgr, dtype=np.uint8)),
            t=int(t),
            received_monotonic_ns=int(received_monotonic_ns),
        )

    def get(self) -> "BoundaryFrameData | None":
        return self._latest


def build_frame_data(
    head_bgr: np.ndarray,
    t: int,
    packed_stereo: bool = True,
    received_monotonic_ns: int = 0,
) -> BoundaryFrameData:
    """boundary head 画像 (BGR) → FrameData。

    orchestrator は head を packed stereo (左右連結) と想定し、perception には左眼、
    policy には左右を渡す。boundary は単一 head なので、packed_stereo=True の時は
    横に複製して packed 幅にする (head_perception_view='left' で左半分=元画像が使われる)。
    """
    img = np.ascontiguousarray(np.asarray(head_bgr, dtype=np.uint8))
    if packed_stereo:
        img = np.concatenate([img, img], axis=1)  # (H, 2W, 3)
    return BoundaryFrameData(
        rgb=img, t=int(t), received_monotonic_ns=int(received_monotonic_ns)
    )


def assemble_19d(
    waist3: np.ndarray | None,
    arms14: np.ndarray,
    hand2: np.ndarray | None,
    *,
    measured_waist3: np.ndarray | None = None,
    fallback_hand2: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """捕捉した各 actuator target → 19D (waist3 + arms14 + hand2)。

    **中身は自前経路の `assemble_action19` に委譲する。** 同じ計算を 2 か所に持つと
    片方だけ直る (実際、自前経路には `measured_waist3` / `fallback_hand2` の
    fallback があるのに、こちらは 0 埋めのままだった)。

    0 埋めの何が悪いか:

    - hand が未 dispatch の tick (deferred model の load 直後など `step()` が
      `None` を返す tick) に `0.0` を入れると、`dex1_model_to_taskspace(0.0)` は
      **`+1.0` = 全閉**。「両手を閉じろ」を運営へ 1 tick 出すことになる
    - waist が未 dispatch の skill (`rotate_table_base` は `dispatch_waist: false`)
      では腰が 0 のまま FK に入り、実際の腰角と違う姿勢で EE を計算して渡す

    Args:
        waist3 / arms14 / hand2: 捕捉した直近 target。未捕捉なら None。
        measured_waist3: `waist3` が None のときに使う実測腰角。
        fallback_hand2: `hand2` が None のときの値 (開始 skill の frame-0 開度)。
    """
    # vendor tree は driver の __init__ で sys.path に入るので、module 直下では
    # import できない。呼ばれた時点では入っている。
    from inference.desktop.lower_policy.actuators.boundary_sink import (
        assemble_action19,
    )

    return assemble_action19(
        waist3,
        np.asarray(arms14, np.float64).reshape(-1)[:14],
        hand2,
        measured_waist3=measured_waist3,
        fallback_hand2=fallback_hand2,
    )
