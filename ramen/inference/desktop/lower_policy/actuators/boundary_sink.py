"""Orchestrator の 19-D action を大会 boundary (`:5556`) へ出す sink。

# なぜ要るか

自前経路 (`entrypoint.py`) は action を `rt/arm_sdk` へ直接 publish する。腕と歩行は
それで動くが、**グリッパは動かない** — `rt/dex1/*/cmd` を購読する serial↔DDS 中継
(`dex1_1_gripper_server`) は我々のラボ構成にしか無く、会場には存在しない
(`docs/setup/thor_pc2_environment.md` §9.6)。

会場でグリッパまで動かせる経路は 1 本だけで、それが boundary:

    我々 → (T,25) を :5556 に publish → 運営 wbc_adapter (pure relay) → WBC → ロボット

`WBC_RUNBOOK` §7 が "wbc_driver.py's gripper path is a verified pure relay (raw values
in, raw values out)" と明記しているとおり、hand 列もそのまま流れる。

# 何を自前で書かないか

publish 側 (bind / 検証 / フレーミング) は **運営の実装をそのまま使う**
(`inference/desktop/boundary/actions.py:DecoupledSink`、vendor・無改変)。
同梱の README が "a local edit here will pass your tests and fail on the robot" と
警告している当のものなので、同等品を自作しない。

自前で持つのは **19-D → 38-D の組み立てだけ**で、そこから先の 38-D → (T,25) は
既存の `taskspace_adapter.groot_chunk_to_taskspace()` を通す。

# 2 つの lane (`lane`、起動の `--boundary-lane`)

- `pose` (既定): (T,25) の手先の姿勢。腕の関節角を FK で手先にし、運営 IK が関節角に戻す。
- `joint` (運営が 2026-09-25 に追加): (T,22) の腕の関節角をそのまま `JointSink` で送る。
  運営 IK を通らない (運営 RUNBOOK: 関節角で学習した policy はこちらが忠実)。手と骨盤高さの列は
  pose と同じ変換 (`dex1_model_to_taskspace` / `check_base_height`) で作る。胴の列は無い
  (どちらの lane でも運営は腰を実測で保持する)。位置の範囲 (URDF) と速さの制限は運営 adapter が掛ける。

# 注意

- **`:5556` は client が bind する側。** 運営の adapter がこちらへ dial-in する
  (`boundary/actions.py` が "THE CLIENT BINDS ... trips everyone up once" と警告)。
  本 sink を Thor 上で動かす場合、adapter の `--actions-host` をそちらへ向けてもらう
  必要がある (**2026-09-20 時点で未検証**)。
- `ee_frame_transform` は既定 `None` (= pelvis/root-link frame)。運営 IK が期待する
  EE 原点が未確定のため (2026-09-15 に質問済み・未回答)。判明したら値を渡すだけ。
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from inference.desktop.lower_policy.policies.taskspace_adapter import (
    DEFAULT_BASE_HEIGHT_M,
    GROOT_ACTION_DIM,
    check_base_height,
    dex1_model_to_taskspace,
    groot_chunk_to_taskspace,
)
from inference.desktop.lower_policy.skills.vla_skill import (
    ACTION_DIM_TOTAL,
    ARMS_SLICE,
    HAND_SLICE,
    WAIST_SLICE,
)

# G1 canonical body order の脚 12 dof (G1JointIndex 0..11)。腰・腕は action で
# 上書きするが、脚は policy が出さないので **実測値をそのまま使う**。
LEG_DOF = 12
BODY_DOF = 29

#: 会場の boundary で使える lane (`BoundaryActionSink(lane=...)` / `--boundary-lane`)
BOUNDARY_LANES = ("pose", "joint")

# EE の計算に使う腰角を更新する閾値 [rad] (Issue #164)。
#
# 運営 IK (`wbc_adapter/ik.py` の `PinkArmIK.solve`) は、目標が前回と完全に同じ
# (`atol=1e-9`) なら解き直さずに前回の関節解を返す。解き直すと実測から seed し直すので、
# 7 自由度の余り (肘の向き) が 1 回ごとに少しずつ流れる (運営自身が記録した drift)。
# 実測の腰角は止まっていても tick ごとに 1e-5〜1.5e-4 rad 揺れる (運営機の実機 capture、
# 180 s / 7012 sample)。それをそのまま FK に入れると腕の指令が一定でも目標が毎回変わり、
# この cache が一度も効かない。模擬の閉ループでは、歩行中の保持で肘が 1.06 → 1.25 rad
# 流れて歩行の範囲の検査で止まった。
#
# 腰は会場では指令しない (運営 WBC が保持) ので、動いたときだけ追従すれば足りる。
# 0.01 rad の遅れで手首の目標は最大 約 3 mm (腰から手首 約 0.3 m) ずれる。実機の run では
# 180 s で腰 yaw が 0.036 rad 動いたので、1 run に数回だけ更新される。
EE_WAIST_REFRESH_RAD = 0.01

# 運営 IK が解く腕の可動域 [rad] (Issue #164)。順は腕 14-D (左 7 + 右 7、
# shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)。
#
# URDF (`assets/organizer_ik/g1_29dof_with_hand.urdf`) の limit に、運営 `ik.py` の
# `IKSettings` が solver 側で上書きする 2 つを重ねたもの:
#     elbow_upper_limit_override = 1.4    (URDF は 2.0944)
#     wrist_roll_limit_override  = 0.9    (URDF は ±1.9722)
# この外の関節角を FK した EE は、運営 IK では残差 1 mm 以内に解けないことがあり、
# そのとき運営 adapter は **その腕の 7 関節すべて**を直前の姿勢で止める。模擬の閉ループ
# (Stage 2) では rotate_leg_to_tighten の左手首 roll が -0.9 に張り付き、左腕の IK が
# 25% 失敗した (教師の左 wrist_roll は -0.99 まで使う)。
# publish の前にこの範囲へ寄せれば、その関節だけが端で止まり、残りは policy どおり動く。
_ORGANIZER_ARM_LOWER = (
    -3.0892, -1.5882, -2.618, -1.0472, -0.9, -1.614429558, -1.614429558,
    -3.0892, -2.2515, -2.618, -1.0472, -0.9, -1.614429558, -1.614429558,
)
_ORGANIZER_ARM_UPPER = (
    2.6704, 2.2515, 2.618, 1.4, 0.9, 1.614429558, 1.614429558,
    2.6704, 1.5882, 2.618, 1.4, 0.9, 1.614429558, 1.614429558,
)
# 端ちょうどは solver の制約に当たって収束が遅くなるので、少し内側を目標にする。
ORGANIZER_IK_LIMIT_MARGIN_RAD = 0.02
ORGANIZER_IK_ARM_LOWER_RAD = np.asarray(_ORGANIZER_ARM_LOWER) + ORGANIZER_IK_LIMIT_MARGIN_RAD
ORGANIZER_IK_ARM_UPPER_RAD = np.asarray(_ORGANIZER_ARM_UPPER) - ORGANIZER_IK_LIMIT_MARGIN_RAD

# 手首 roll の上書き (0.9) は運営 IK の a1af470 (interface package 609f61d) で無くなった
# (姿勢の重みで寄せる形に変わり、上限は URDF の ±1.9722 だけ)。`clamp_wrist_roll=False`
# (起動の `--wrist-roll-clamp off`) では手首 roll をこの URDF の範囲にだけ収める。
# 古い運営 IK (0.9 が固い上限) の相手に外すと、0.9 を超えた目標で腕ごと止まる。
WRIST_ROLL_INDICES = (4, 11)
URDF_WRIST_ROLL_LIMIT_RAD = 1.972222054


def clamp_arms_to_organizer_ik(
    arms14: Sequence[float], *, clamp_wrist_roll: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """腕 14-D を運営 IK の可動域 (余裕つき) に収める。

    Args:
        clamp_wrist_roll: False なら手首 roll は 0.9 ではなく URDF の範囲に収める。

    Returns:
        (収めた腕 14-D, 端に寄せた関節の bool mask)。
    """
    arms = np.asarray(arms14, dtype=np.float64).reshape(-1)
    if arms.shape != (14,):
        raise ValueError(f"arms14 must be (14,), got {arms.shape}")
    lower = ORGANIZER_IK_ARM_LOWER_RAD.copy()
    upper = ORGANIZER_IK_ARM_UPPER_RAD.copy()
    if not clamp_wrist_roll:
        for index in WRIST_ROLL_INDICES:
            lower[index] = -URDF_WRIST_ROLL_LIMIT_RAD + ORGANIZER_IK_LIMIT_MARGIN_RAD
            upper[index] = URDF_WRIST_ROLL_LIMIT_RAD - ORGANIZER_IK_LIMIT_MARGIN_RAD
    clamped = np.clip(arms, lower, upper)
    return clamped, clamped != arms


def waist_joints_to_torso_rpy(waist_yaw_roll_pitch: Sequence[float]) -> np.ndarray:
    """Convert the G1 yaw->roll->pitch serial chain to standard XYZ RPY.

    The three G1 waist joint values are *not* generally the Euler angles of the
    torso.  The URDF composes ``Rz(yaw) @ Rx(roll) @ Ry(pitch)``, whereas the
    organizer's torso RPY command represents ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    They coincide for one-axis motion but diverge when roll and pitch are both
    non-zero, which also makes the WBC torso frame disagree with the frame used
    for the wrist FK.  Convert through the actual rotation matrix.
    """

    q = np.asarray(waist_yaw_roll_pitch, dtype=np.float64).reshape(-1)
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("waist_yaw_roll_pitch must be finite 3-D")
    yaw, roll, pitch = (float(v) for v in q)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.asarray(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.asarray(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rotation = rz @ rx @ ry

    # Inverse of Rz(yaw) @ Ry(pitch) @ Rx(roll), away from the unreachable
    # gimbal singularity (G1 waist pitch is mechanically limited to +/-0.52).
    standard_pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    standard_roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    standard_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return np.asarray(
        [standard_roll, standard_pitch, standard_yaw], dtype=np.float64
    )


class BoundaryWalkActuator:
    """Navigation command holder for the decoupled 25-D boundary row.

    This deliberately has no Unitree SDK client.  ``set_velocity`` updates the
    boundary command and asks the configured publisher to emit immediately.
    """

    def __init__(self) -> None:
        self.latest = np.zeros(3, dtype=np.float64)
        self._publish: Optional[Callable[[], None]] = None

    def set_publish_callback(self, callback: Callable[[], None]) -> None:
        self._publish = callback

    def set_velocity(
        self, vx: float, vy: float, vyaw: float, *, duration: float | None = None
    ) -> None:
        del duration
        value = np.asarray([vx, vy, vyaw], dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError("boundary navigation command must be finite")
        self.latest = value
        if self._publish is not None:
            self._publish()


class BoundaryArmActuator:
    """Small arm ownership adapter with no DDS publisher.

    It gives the existing safety/transition machinery the same measured-state
    and last-target surface as ``G1ArmActuator`` while every physical command
    is routed exclusively through ``BoundaryActionSink``.
    """

    def __init__(self) -> None:
        self._sender: Optional[Callable[[Sequence[float]], bool]] = None
        self._state_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        # No command has been published yet.  In particular, zero joint angles
        # are never a safe substitute for a missing first state sample.
        self._last: Optional[np.ndarray] = None
        self._initial: Optional[np.ndarray] = None
        self._started = False
        self._has_motion_target = False

    def configure(
        self,
        *,
        sender: Callable[[Sequence[float]], bool],
        state_getter: Callable[[], Optional[np.ndarray]],
    ) -> None:
        self._sender = sender
        self._state_getter = state_getter
        # State may not have arrived during construction. start() re-reads it
        # after the final preflight/operator gate.
        self._last = None
        self._initial = None
        self._has_motion_target = False

    def start(self) -> None:
        if self._state_getter is None or self._sender is None:
            raise RuntimeError("boundary arm actuator is not configured")
        measured = self._state_getter()
        if measured is None:
            raise RuntimeError("fresh measured arm state unavailable at boundary start")
        measured = np.asarray(measured, dtype=np.float64).reshape(-1)
        if measured.shape != (14,) or not np.all(np.isfinite(measured)):
            raise RuntimeError("boundary start requires finite measured arm state (14-D)")
        self._last = measured.copy()
        self._initial = measured.copy()
        self._has_motion_target = False
        self._started = True

    def send_action(self, arms14: Sequence[float]) -> None:
        """腕 14-D を boundary へ出す。**唯一の送信窓口**。

        「最後に送った target」は publish に成功したときだけ更新する。歩行指令の
        再送・model 遷移の起点・終了時の保持はこの値を読むので、publish されて
        いない target (関節 state 未受信の tick、検証で弾かれた row) を記録すると、
        実機が一度も向かっていない姿勢へ跳ぶ。起動前 (`start` 前) は publish
        しないので記録もしない。
        """
        target = np.asarray(arms14, dtype=np.float64).reshape(-1)
        if target.shape != (14,) or not np.all(np.isfinite(target)):
            raise ValueError("boundary arm target must be finite 14-D")
        if not self._started:
            return
        if self._sender is None:
            raise RuntimeError("boundary arm publisher is not configured")
        if self._sender(target.copy()):
            if self._initial is not None and np.max(np.abs(target - self._initial)) > 0.01:
                self._has_motion_target = True
            self._last = target.copy()

    @property
    def has_motion_target(self) -> bool:
        """Whether an arm target distinct from the initial hold was published."""
        return self._has_motion_target

    def read_arm_positions(self) -> np.ndarray:
        if self._state_getter is None:
            raise RuntimeError("boundary state source is not configured")
        measured = self._state_getter()
        if measured is None:
            raise RuntimeError("boundary state is unavailable")
        return np.asarray(measured, dtype=np.float64).copy()

    def read_last_published_targets(self):
        if self._last is None:
            raise RuntimeError("boundary arm hold target is unavailable before start")
        return self._last.copy(), None

    def clear_waist_action(self) -> None:
        return None

    def stop(self) -> None:
        self._started = False


def assemble_action38(
    action19: Sequence[float],
    body_q29: Sequence[float],
) -> np.ndarray:
    """19-D action + 実測 body_q(29) → GR00T raw 38-D (root7 + body29 + hand2)。

    提出側 `components/ramen/orchestrator_driver.py` が同じ組み立てをしている。
    FK は root7 を使わない (pelvis 基準の相対 chain) ので、root は identity で埋める。

    Args:
        action19: `vla_skill` の 19-D (waist3 + arms14 + hand2)。
        body_q29: `rt/lowstate` 由来の実測関節角 (G1JointIndex 順)。脚 12 dof のみ使う。

    Returns:
        (38,) float64。`groot_chunk_to_taskspace` にそのまま渡せる。
    """

    action = np.asarray(action19, dtype=np.float64).reshape(-1)
    if action.shape != (ACTION_DIM_TOTAL,):
        raise ValueError(f"action19 must be ({ACTION_DIM_TOTAL},), got {action.shape}")
    body = np.asarray(body_q29, dtype=np.float64).reshape(-1)
    if body.shape != (BODY_DOF,):
        raise ValueError(f"body_q29 must be ({BODY_DOF},), got {body.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("action19 must be finite")
    if not np.all(np.isfinite(body[:LEG_DOF])):
        raise ValueError("body_q29 legs must be finite")

    # root7 = 位置 0 + 単位 quat (w-first)。FK が使わないので値は効かないが、
    # 38-D の形を崩さないために埋める。
    root = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    body29 = np.concatenate([body[:LEG_DOF], action[WAIST_SLICE], action[ARMS_SLICE]])
    return np.concatenate([root, body29, action[HAND_SLICE]])


def assemble_action19(
    waist3: Optional[Sequence[float]],
    arms14: Sequence[float],
    hand2: Optional[Sequence[float]],
    *,
    measured_waist3: Optional[Sequence[float]] = None,
    fallback_hand2: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    """腰 / 腕 / 手の 3 断片から 19-D action を組み直す。

    orchestrator が `actuator_send_fn` に渡すのは **腕 14-D だけ**で、腰と手は skill が
    それぞれの actuator へ直接送っている。boundary へ出すには 19-D 全体が要るので、
    mock actuator が保持している直近 target (`latest`) から組み直す
    (提出側 `orchestrator_driver.py` の `assemble_19d` と同じ考え方)。

    Args:
        waist3: 腰 actuator の直近 target。skill がまだ送っていなければ `None`。
        arms14: orchestrator から来た腕 14-D。
        hand2: 手 actuator の直近 target。未送信なら `None`。
        measured_waist3: `waist3` が `None` のときに使う実測腰角。これも無ければ 0。
        fallback_hand2: `hand2` が `None` のときの値。
    """

    arms = np.asarray(arms14, dtype=np.float64).reshape(-1)
    if arms.shape != (14,):
        raise ValueError(f"arms14 must be (14,), got {arms.shape}")

    if waist3 is not None:
        waist = np.asarray(waist3, dtype=np.float64).reshape(-1)
    elif measured_waist3 is not None:
        waist = np.asarray(measured_waist3, dtype=np.float64).reshape(-1)
    else:
        waist = np.zeros(3, dtype=np.float64)
    if waist.shape != (3,):
        raise ValueError(f"waist must be (3,), got {waist.shape}")

    hand = np.asarray(
        hand2 if hand2 is not None else fallback_hand2, dtype=np.float64
    ).reshape(-1)
    if hand.shape != (2,):
        raise ValueError(f"hand must be (2,), got {hand.shape}")

    return np.concatenate([waist, arms, hand])


def action38_to_joint_row(
    action_38: Sequence[float],
    *,
    navigate_cmd: Sequence[float] = (0.0, 0.0, 0.0),
    base_height_cmd: float = DEFAULT_BASE_HEIGHT_M,
) -> np.ndarray:
    """38-D action 1 row → joint lane の (22,) row (運営 `JointSink.make_rows` で並べる)。

    腕は body[15:29] (左 7 + 右 7、Unitree 順 = :5557 の body_q と同じ) をそのまま。手と骨盤高さは
    (T,25) と同じ変換を通すので、どちらの lane でも同じ値が届く。`make_rows` は高さを省くと 0 m
    (骨盤を床へ) にするので、必ず渡す。
    """
    from inference.desktop.boundary import JointSink

    a = np.asarray(action_38, dtype=np.float64).reshape(-1)
    if a.shape != (GROOT_ACTION_DIM,):
        raise ValueError(f"action must be ({GROOT_ACTION_DIM},), got {a.shape}")
    body29 = a[7:36]
    left_hand = dex1_model_to_taskspace(float(a[36]))
    right_hand = dex1_model_to_taskspace(float(a[37]))
    rows = JointSink.make_rows(
        left_arm=body29[15:22],
        right_arm=body29[22:29],
        left_hand=[left_hand, left_hand],  # 2 指に同値 ((T,25) と同じ)
        right_hand=[right_hand, right_hand],
        navigate=np.asarray(navigate_cmd, dtype=np.float64).reshape(1, 3),
        base_height=[check_base_height(base_height_cmd)],
    )
    return rows[0]


class BoundaryActionSink:
    """19-D action を (T,25) (pose lane) か (T,22) (joint lane) にして運営 boundary へ publish する。

    `rt/arm_sdk` 直の actuator と**同時には使わない**。両方が同じ関節を動かすため。

    Args:
        fk: `G1WristFK` 相当 (`compute_ee_transforms` を持つもの)。省略時は運営 IK と
            同じ運動学 (`ORGANIZER_IK_URDF_PATH`) で作る。会場では省略すること
            (学習用の mode_15 URDF を渡すと手首が約 5 mm ずれる、Issue #164)。joint lane では使わない。
        lane: `"pose"` (既定、`DecoupledSink`) か `"joint"` (`JointSink`)。module の docstring。
        clamp_wrist_roll: pose lane で手首 roll を運営 IK の古い上限 0.9 に寄せるか
            (`clamp_arms_to_organizer_ik`)。joint lane は寄せない (運営 IK を通らない)。
        port / host: `DecoupledSink` にそのまま渡す。
        ee_frame_transform: root-link → 運営 IK が期待する frame の 4x4。未確定なので
            既定 `None` (変換なし)。
        log_fn: `dict` を 1 件受け取る callable。publish した raw (T,25) を残す
            (`WBC_RUNBOOK` §5:「グリッパが閉じたかは自分の publish 値で分かる」)。
        sender_clock_offset_fn: 運営 bridge (PC2) の壁時計 − この host の壁時計 [s] を
            返す callable (`ZmqFrameSource.sender_clock_offset_s`)。運営 adapter は
            `issued_at` を PC2 の時計と比べて 1.0 s より古い chunk を捨てるので、
            Thor で動かすときは送信時刻を PC2 の時計に揃える (Issue #161、C2-01)。
            None を返す間 (まだカメラが来ていない) は今までどおりこの host の時計。
    """

    def __init__(
        self,
        fk: Any = None,
        *,
        lane: str = "pose",
        clamp_wrist_roll: bool = True,
        port: int = 5556,
        host: str = "*",
        ee_frame_transform: Optional[np.ndarray] = None,
        log_fn: Any = None,
        sender_clock_offset_fn: Optional[Callable[[], Optional[float]]] = None,
    ) -> None:
        if lane not in BOUNDARY_LANES:
            raise ValueError(f"lane must be one of {BOUNDARY_LANES}, got {lane!r}")
        if lane == "pose":
            if fk is None:
                from inference.desktop.perception.g1_urdf_fk import (
                    ORGANIZER_IK_URDF_PATH,
                    G1WristFK,
                )

                fk = G1WristFK.from_urdf(ORGANIZER_IK_URDF_PATH)
            if not callable(getattr(fk, "compute_ee_transforms", None)):
                raise ValueError(
                    "official decoupled boundary requires the G1 URDF wrist FK; "
                    "refusing to publish zero/undefined end-effector poses"
                )
        # boundary は zmq / cv2 / msgpack を引くので lazy import
        # (default env から本 module を import しても壊さない)。lane で使う方だけ。
        if lane == "pose":
            from inference.desktop.boundary import DecoupledSink as sink_class
        else:
            from inference.desktop.boundary import JointSink as sink_class

        self._lane = lane
        self._clamp_wrist_roll = bool(clamp_wrist_roll)
        self._fk = fk if lane == "pose" else None
        self._ee_frame_transform = ee_frame_transform
        self._log_fn = log_fn
        self._sender_clock_offset_fn = sender_clock_offset_fn
        self._sink = sink_class(port=port, host=host)
        self._sent = 0
        # EE の計算に使っている腰角 (EE_WAIST_REFRESH_RAD 以上動いたときだけ更新)。
        self._ee_waist: Optional[np.ndarray] = None
        # 運営 IK の可動域へ寄せた回数 (clamp_arms_to_organizer_ik)。
        self._clamped = 0
        print(
            f"[boundary] {type(self._sink).__name__} ({lane} lane) bound on {host}:{port} "
            "(the organizer's adapter dials in to this)",
            file=sys.stderr,
        )
        if lane == "pose":
            # 接続テストで決める設定なので、どちらで動いているかを起動 log に残す
            print(
                "[boundary] wrist_roll clamp: "
                + (
                    "on (organizer IK 0.9 cap)"
                    if self._clamp_wrist_roll
                    else "off (URDF limit only; organizer IK a1af470+)"
                ),
                file=sys.stderr,
            )

    @property
    def lane(self) -> str:
        return self._lane

    @property
    def sent_count(self) -> int:
        return self._sent

    def send_action(
        self,
        action19: Sequence[float],
        body_q29: Sequence[float],
        *,
        navigate_cmd: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        """1 tick 分の 19-D action を (1,25) (pose) か (1,22) (joint) chunk として publish する。"""

        navigation = np.asarray(navigate_cmd, dtype=np.float64).reshape(-1)
        if navigation.shape != (3,) or not np.all(np.isfinite(navigation)):
            raise ValueError("navigate_cmd must be finite 3-D")
        if self._lane == "joint":
            chunk, record = self._joint_chunk(action19, body_q29, navigation)
        else:
            chunk, record = self._pose_chunk(action19, body_q29, navigation)
        # 送信時刻を運営 adapter の時計 (PC2) に揃える。推定がまだ無ければ
        # sink の既定 (この host の time.time()) のまま。
        offset = (
            None
            if self._sender_clock_offset_fn is None
            else self._sender_clock_offset_fn()
        )
        issued_at = None if offset is None else time.time() + float(offset)
        # 検証は sink.send_chunk が中でやる (不正なら ActionError)。
        if issued_at is None:
            self._sink.send_chunk(chunk)
        else:
            self._sink.send_chunk(chunk, issued_at=issued_at)
        self._sent += 1
        if self._log_fn is not None:
            self._log_fn(
                {
                    "seq": self._sent,
                    **record,
                    # None = この host の時計で付けた (送信側の時計の差がまだ無い)。
                    "issued_at": issued_at,
                    "sender_clock_offset_s": offset,
                    # 逆算用: この tick で読めた実測関節角。静止保持中の後半を使えば
                    # 「指令した EE」と「到達した関節を自前 FK に通した EE」の差 =
                    # 運営 IK が期待する frame とのオフセットが解ける (EE 原点が
                    # どの資料にも無いため、実測から求めるしかない)。
                    "measured_body_q29": np.asarray(
                        body_q29, dtype=np.float64
                    ).tolist(),
                }
            )

    def _pose_chunk(
        self,
        action19: Sequence[float],
        body_q29: Sequence[float],
        navigation: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """pose lane: (1,25) の手先の姿勢の chunk と、log に足す項目。"""
        action19 = self._stabilize_waist(action19)
        # 運営 IK が解ける範囲へ。外れた関節があると運営 adapter がその腕を丸ごと止める。
        arms, clamped_mask = clamp_arms_to_organizer_ik(
            action19[ARMS_SLICE], clamp_wrist_roll=self._clamp_wrist_roll
        )
        action19[ARMS_SLICE] = arms
        if clamped_mask.any():
            self._clamped += 1
        action38 = assemble_action38(action19, body_q29)
        chunk = groot_chunk_to_taskspace(
            action38[None, :], self._fk, ee_frame_transform=self._ee_frame_transform
        )
        chunk[:, 18:21] = navigation
        # action19 stores the three serial G1 waist joints (yaw->roll->pitch),
        # while the organizer contract wants the resulting torso orientation
        # as standard XYZ RPY.  A reorder is insufficient for compound motion.
        # The entrypoint fills the waist with the *measured* angle (the
        # organizer's adapter ignores [22:25] and holds the measured waist), so
        # these columns report the torso orientation, held within
        # EE_WAIST_REFRESH_RAD of the measurement (_stabilize_waist).
        waist_yaw_roll_pitch = np.asarray(action19, dtype=np.float64)[:3]
        chunk[:, 22:25] = waist_joints_to_torso_rpy(waist_yaw_roll_pitch)
        row = np.asarray(chunk[0], dtype=np.float64)
        return chunk, {
            "event": "boundary_taskspace",
            # hand 列は「掴んだか」の一次証拠。-1=open / +1=closed。
            "left_hand": row[0:2].tolist(),
            "right_hand": row[2:4].tolist(),
            "left_ee_pos": row[4:7].tolist(),
            "right_ee_pos": row[11:14].tolist(),
            "taskspace_25": row.tolist(),
            # 運営 IK の可動域へ寄せた腕の関節 (腕 14-D の index)。空なら無し。
            "organizer_ik_clamped_joints": np.flatnonzero(clamped_mask).tolist(),
        }

    def _joint_chunk(
        self,
        action19: Sequence[float],
        body_q29: Sequence[float],
        navigation: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """joint lane: (1,22) の腕の関節角の chunk と、log に足す項目。

        FK・運営 IK 用の clamp・腰角の据え置きは要らない (どれも運営 IK のため)。
        """
        action38 = assemble_action38(
            np.asarray(action19, dtype=np.float64).reshape(-1).copy(), body_q29
        )
        row = action38_to_joint_row(action38, navigate_cmd=navigation)
        return row[None, :], {
            "event": "boundary_joint",
            # hand 列は「掴んだか」の一次証拠。-1=open / +1=closed。
            "left_hand": row[0:2].tolist(),
            "right_hand": row[2:4].tolist(),
            "joint_22": row.tolist(),
        }

    def _stabilize_waist(self, action19: Sequence[float]) -> np.ndarray:
        """腰角 [0:3] を、`EE_WAIST_REFRESH_RAD` 以上動くまで前回の値に据え置く。

        腕と手の指令が同じなら publish する行が bit 単位で同じになり、運営 IK の
        「同じ目標は解き直さない」cache が効く (`EE_WAIST_REFRESH_RAD` の説明)。
        """
        action = np.asarray(action19, dtype=np.float64).reshape(-1).copy()
        waist = action[WAIST_SLICE]
        if not np.all(np.isfinite(waist)):
            # 検証は assemble_action38 に任せる (ここで握りつぶさない)。
            return action
        if (
            self._ee_waist is None
            or float(np.max(np.abs(waist - self._ee_waist))) > EE_WAIST_REFRESH_RAD
        ):
            self._ee_waist = waist.copy()
        action[WAIST_SLICE] = self._ee_waist
        return action

    def close(self) -> None:
        """冪等。"""

        sink, self._sink = self._sink, None
        if sink is not None:
            sink.close()
