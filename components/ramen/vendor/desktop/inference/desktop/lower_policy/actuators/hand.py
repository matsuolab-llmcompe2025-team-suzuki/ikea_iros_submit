"""Hand (Dex1-1) actuator interface + Mock 実装 + G1 real 実装 (Issue #125 Phase 9a / B-2/B-3)。

RAMEN-Ori / GR00T の action 19D の末尾 2D = hand grip position (left / right)。
物理的には Dex1-1 (Unitree の Dex hand) の 5-finger control が絡むが、training
の action space は 2D (left_gripper_q, right_gripper_q) の simple grip signal
に集約されている (subtask_training.json:state.names 末尾 2 と action.names 末尾 2)。

# Dex1-1 DDS pipeline (docs/inference/apple_vision_pro_upper_body_teleop.md §4.4)

- **Topic**: `rt/dex1/left/cmd` / `rt/dex1/right/cmd` (per-hand)
    * G1 の rt/arm_sdk とは別系統 = Orin 上の `dex1_1_gripper_server` daemon が
      USB シリアル ↔ DDS を中継する (systemd で自動起動)
- **State topic** (parallel): `rt/dex1/left/state` / `rt/dex1/right/state`
    * `MotorStates_.states[0].q` で現在 gripper q が取れる (check_dex1_state.py 参照)
- **Message type**: `MotorCmds_` (unitree_go.msg.dds_) = `cmds: sequence[MotorCmd_]`
    * `cmds[0].q` = 目標 gripper position (open/close の single scalar)
    * training data 表現とも整合 (g1_full_body_mapping.map_source_row_to_real_g1_relative_eef
      の Dex1 hand pack: "one open/close coordinate per hand、first element only")
- **Frequency**: 200 Hz (公式Dex1 test controllerと同じ5 ms周期)

# 現状

- **Mock 実装** (dev / test 用、SDK 依存無し)
- **G1HandActuator real 実装** (unitree_sdk2py.idl.unitree_go.msg.dds_.MotorCmds_
  経由で dex1_1_gripper_server topic に publish)
- Orin 側 daemon (`dex1_1_gripper_server.service`) が動いていること、Dex1-1
  hardware が USB 接続で認識されていることが実運用の前提 (docs §4.4 参照)

# Signal semantics

Training-side は `left_gripper_q` / `right_gripper_q` を物理 motor-output rad
として保持する。公式 server もDDS ``q``を無変換で使う。0 radが閉側、約5.4 rad
が開側であり、学習データの command は通常0--4.5 rad。範囲外はhardware envelope
の0--5.4 radへclampする。

# Protocol lifecycle

`send_action` + `start` + `stop` の 3 method。start は SDK Publisher init +
RecurrentThread (100 Hz) start、stop は thread join。Mock は no-op。
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np


G1_NUM_HAND_JOINTS: int = 2  # left_gripper_q, right_gripper_q
HAND_GRIP_MIN: float = 0.0   # 閉側hardware下限 [rad]
HAND_GRIP_MAX: float = 5.4   # 公式Dex1 calibrationに基づく開側hardware上限 [rad]

# Dex1-1 SDK topic 名 (docs/inference/apple_vision_pro_upper_body_teleop.md §4.4)。
G1_DEX1_LEFT_CMD_TOPIC: str = "rt/dex1/left/cmd"
G1_DEX1_RIGHT_CMD_TOPIC: str = "rt/dex1/right/cmd"
G1_DEX1_LEFT_STATE_TOPIC: str = "rt/dex1/left/state"
G1_DEX1_RIGHT_STATE_TOPIC: str = "rt/dex1/right/state"
G1_DEX1_CONTROL_FREQ_HZ: float = 200.0
G1_DEX1_POSITION_KP: float = 5.0
G1_DEX1_POSITION_KD: float = 0.05
# Dex1-1 hand の motor 数 (dex1_1_gripper_server が expose する)。Dex1-1 は
# open/close single-motor per hand、cmds sequence 長 = 1。
G1_DEX1_NUM_MOTORS_PER_HAND: int = 1

# 後方互換 alias (旧 skeleton から参照している場合の grace period、次期 clean up 予定)
G1_DEX1_LEFT_TOPIC: str = G1_DEX1_LEFT_CMD_TOPIC
G1_DEX1_RIGHT_TOPIC: str = G1_DEX1_RIGHT_CMD_TOPIC
G1_DEX1_STATE_TOPIC: str = G1_DEX1_LEFT_STATE_TOPIC  # 単一 topic を期待していた旧参照向け


class HandActuator(Protocol):
    """Hand 2-joint (left/right gripper) action sink。

    Lifecycle: init → start (publish thread 起動) → 複数回 send_action → stop。
    Mock は start/stop no-op、real 実装は SDK publisher thread の制御に使う。
    """

    def send_action(self, positions: Sequence[float]) -> None:
        """Send physical Dex1 motor-output position [rad]。

        Args:
            positions: length-2 sequence、(left_gripper_q, right_gripper_q) 順。
                       範囲外は [0.0, 5.4] rad に clamp。
        """
        ...

    def start(self) -> None:
        """Publish thread を起動 (Mock は no-op)。冪等。"""
        ...

    def stop(self) -> None:
        """Publish thread を停止 + resource release (Mock は no-op)。冪等。"""
        ...


class MockHandActuator:
    """DDS 送信せずに直近 target を保持する mock。tests + entrypoint dry-run 用。

    Attributes:
        history: send_action() 呼出履歴 (append-only)。
        latest: 直近の (left, right) or None。
        started: start() が呼ばれたか (lifecycle test 用)。
    """

    def __init__(self) -> None:
        self.history: list[tuple[float, float]] = []
        self.latest: tuple[float, float] | None = None
        self.started: bool = False

    def send_action(self, positions: Sequence[float]) -> None:
        arr = tuple(float(p) for p in positions)
        if len(arr) != G1_NUM_HAND_JOINTS:
            raise ValueError(
                f"hand positions must have length {G1_NUM_HAND_JOINTS}, got {len(arr)}"
            )
        clamped = tuple(max(HAND_GRIP_MIN, min(HAND_GRIP_MAX, v)) for v in arr)
        self.history.append((clamped[0], clamped[1]))
        self.latest = (clamped[0], clamped[1])

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


@dataclass
class _LatestHandTarget:
    """Dex1-1 real hand target (Phase B-3)。arm buffer と同 pattern (lock 保護)。"""

    positions: Optional[tuple[float, float]] = None  # (left, right) physical rad
    received: bool = False


class G1HandActuator:
    """G1 Dex1-1 hand real actuator (Phase B-3)。

    ## 実 pipeline (docs/inference/apple_vision_pro_upper_body_teleop.md §4.4)

    - Desktop → DDS (`rt/dex1/{left,right}/cmd`, MotorCmds_) → Orin `dex1_1_gripper_server`
      daemon → USB シリアル → Dex1-1 hardware。
    - Publish 頻度 = 200 Hz (`G1_DEX1_CONTROL_FREQ_HZ`)、`MotorCmds_.cmds[0].q` に
      physical motor-output rad を書く。公式dex1_1_gripper_serverはDDS値を
      無変換でmotorへ渡す。
    - training-side hand_state (2D) の pack convention と整合: "one open/close
      coordinate per hand、first element only" (g1_full_body_mapping.py L253-256)。

    ## 前提

    - **Orin 側 `dex1_1_gripper_server.service` が起動していること** (docs §4.4)。
      起動していないと Desktop 側 publish は成功するが、hardware に届かない。
    - Dex1-1 が USB で Orin に接続されていること。
    - `unitree_sdk2py` (cyclonedds 経由) が runtime env に install 済 (arm と共通)。
    - ChannelFactory が既に init 済 (G1SDKWalkActuator or G1ArmActuator が先に呼ぶ)。

    ## Interface

    - `send_action(positions_2d)`: (left_q, right_q) [rad] をbufferに格納、clamp [0, 5.4]
    - `start()`: SDK Publisher × 2 (left/right) init + RecurrentThread 200Hz start
    - `stop()`: thread stop
    - unit test は依存注入で real SDK 不要 (publisher/thread mock でも同じ path)

    Args:
        control_freq_hz: publish 頻度 [Hz]。default 100Hz (docs 準拠)。
        left_publisher / right_publisher: 依存注入 (test 用)。None なら start() で
            SDK 経由に init する。
        publish_thread: 依存注入 (test 用)。None なら SDK RecurrentThread。
        cmd_factory: MotorCmds_ instance を生成する callable (test 用に inject 可)。
            None なら SDK から MotorCmds_ / MotorCmd_ を import。
    """

    def __init__(
        self,
        control_freq_hz: float = G1_DEX1_CONTROL_FREQ_HZ,
        velocity_limit_rad_s: Optional[float] = None,
        left_publisher: Optional[object] = None,
        right_publisher: Optional[object] = None,
        publish_thread: Optional[object] = None,
        cmd_factory: Optional[object] = None,
    ) -> None:
        self._control_freq_hz = float(control_freq_hz)
        if velocity_limit_rad_s is not None and velocity_limit_rad_s <= 0.0:
            raise ValueError("velocity_limit_rad_s must be positive when provided")
        self._velocity_limit_rad_s = (
            float(velocity_limit_rad_s)
            if velocity_limit_rad_s is not None
            else None
        )
        self._lock = threading.Lock()
        self._target = _LatestHandTarget()
        self._commanded_positions: Optional[tuple[float, float]] = None
        self._running = False
        self._left_publisher = left_publisher
        self._right_publisher = right_publisher
        self._publish_thread = publish_thread
        self._cmd_factory = cmd_factory  # test-injected MotorCmds_ factory

    def send_action(self, positions: Sequence[float]) -> None:
        arr = tuple(float(p) for p in positions)
        if len(arr) != G1_NUM_HAND_JOINTS:
            raise ValueError(
                f"hand positions must have length {G1_NUM_HAND_JOINTS}, got {len(arr)}"
            )
        clamped = tuple(max(HAND_GRIP_MIN, min(HAND_GRIP_MAX, v)) for v in arr)
        with self._lock:
            self._target.positions = (clamped[0], clamped[1])
            self._target.received = True

    def prime_hold(self, positions: Sequence[float]) -> None:
        """Seed the real measured pose before starting a slew-limited publisher.

        This is used by the Phase 1.13 gate after live Dex1 state is available.
        The first DDS write therefore holds the measured pose rather than
        jumping directly to the first learned-policy target.
        """

        arr = tuple(float(p) for p in positions)
        if len(arr) != G1_NUM_HAND_JOINTS:
            raise ValueError(
                f"hand positions must have length {G1_NUM_HAND_JOINTS}, got {len(arr)}"
            )
        clamped = tuple(max(HAND_GRIP_MIN, min(HAND_GRIP_MAX, v)) for v in arr)
        with self._lock:
            self._commanded_positions = (clamped[0], clamped[1])
            self._target.positions = (clamped[0], clamped[1])
            self._target.received = True

    def start(self) -> None:
        """Publish thread を起動。DI 未注入なら SDK Publisher init。"""
        with self._lock:
            if self._running:
                return
            self._running = True
        if self._left_publisher is None or self._right_publisher is None:
            self._init_publishers()
        if self._publish_thread is None:
            self._init_publish_thread()
        self._publish_thread.Start()  # type: ignore[attr-defined]

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        if self._publish_thread is not None and hasattr(self._publish_thread, "Wait"):
            self._publish_thread.Wait(timeout=1.0)  # type: ignore[attr-defined]

    def _init_publishers(self) -> None:
        """SDK ChannelPublisher × 2 (left/right) の lazy init。

        SDK は unitree_sdk2py に既に含まれる (dex1 専用 SDK は不要)。ChannelFactory
        は G1SDKWalkActuator or G1ArmActuator が先に init 済想定 (entrypoint 経路)。
        """
        from unitree_sdk2py.core.channel import ChannelPublisher  # type: ignore
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_  # type: ignore

        if self._left_publisher is None:
            self._left_publisher = ChannelPublisher(G1_DEX1_LEFT_CMD_TOPIC, MotorCmds_)
            self._left_publisher.Init()
        if self._right_publisher is None:
            self._right_publisher = ChannelPublisher(G1_DEX1_RIGHT_CMD_TOPIC, MotorCmds_)
            self._right_publisher.Init()
        if self._cmd_factory is None:
            self._cmd_factory = _make_default_cmd_factory()

    def _init_publish_thread(self) -> None:
        from unitree_sdk2py.utils.thread import RecurrentThread  # type: ignore

        self._publish_thread = RecurrentThread(
            interval=1.0 / max(self._control_freq_hz, 1.0),
            target=self._publish_once,
            name="g1_dex1_publish",
        )

    def _publish_once(self) -> None:
        """1 tick 分の (left_q, right_q) を 2 topic に publish (200Hz)。"""
        with self._lock:
            if (
                not self._running
                or not self._target.received
                or self._target.positions is None
            ):
                return
            target = np.asarray(self._target.positions, dtype=np.float64)
            if self._velocity_limit_rad_s is None:
                commanded = target
            else:
                if self._commanded_positions is None:
                    # A production slew-limited path must call prime_hold()
                    # from a measured Dex1 state before start().
                    return
                previous = np.asarray(self._commanded_positions, dtype=np.float64)
                max_step = self._velocity_limit_rad_s / max(
                    self._control_freq_hz, 1.0
                )
                commanded = previous + np.clip(target - previous, -max_step, max_step)
            left, right = float(commanded[0]), float(commanded[1])
            self._commanded_positions = (left, right)
        try:
            if self._cmd_factory is None:
                # DI 経路 (test) で cmd_factory 未注入なら生 tuple を書く
                # (fake publisher が受け取る簡易 pattern)
                self._left_publisher.Write((left,))  # type: ignore[attr-defined]
                self._right_publisher.Write((right,))  # type: ignore[attr-defined]
            else:
                left_cmd = self._cmd_factory()
                right_cmd = self._cmd_factory()
                left_cmd.cmds[0].q = float(left)  # type: ignore[attr-defined]
                right_cmd.cmds[0].q = float(right)  # type: ignore[attr-defined]
                self._left_publisher.Write(left_cmd)  # type: ignore[attr-defined]
                self._right_publisher.Write(right_cmd)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[G1HandActuator] publish error: {exc!r}", file=sys.stderr)


def _make_default_cmd_factory():
    """MotorCmds_ instance (cmds sequence 1 slot 初期化済) を返す factory。

    Dex1-1 は single-motor per hand (`G1_DEX1_NUM_MOTORS_PER_HAND=1`)。SDK の
    `unitree_go_msg_dds__MotorCmds_` default helper は空 sequence を返すため、
    ここで MotorCmd_ を 1 個 append する初期化を hooked in。lazy import。
    """
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_  # type: ignore

    def _factory():
        cmd = MotorCmd_(
            mode=1,
            q=0.0,
            dq=0.0,
            tau=0.0,
            kp=G1_DEX1_POSITION_KP,
            kd=G1_DEX1_POSITION_KD,
            reserve=[0, 0, 0],
        )
        return MotorCmds_(cmds=[cmd])

    return _factory
