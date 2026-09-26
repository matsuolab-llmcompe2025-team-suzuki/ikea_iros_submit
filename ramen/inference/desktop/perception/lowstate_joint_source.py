"""G1 の `rt/lowstate` から関節角を直接受ける source (Issue #141 束 1-14 / INF-18)。

# なぜ要るか

Orin の bridge (`real_hw_bridge_node`) が publish する `/joint_states` は、**同じ値を
新しい `header.stamp` で送り直している**。09-09 の実機 9 run で、繰り返しの sample は
position / velocity / effort が bit 単位で一致し、値が変わるのは 11.4〜13.3 Hz しか
なかった (policy の loop を何 Hz で回しても変わらない)。30 Hz の tick で見ると、
2〜3 tick に 1 回しか新しい関節角が来ないことになる。

Desktop は腕の actuator (`G1ArmActuator`) ですでに `rt/lowstate` を直接受けている。
関節角も同じところから取れば、bridge の詰まりを通らずに済む。

`JointStateSource` と同じ `JointStateData` を返すので、下流 (観測の組み立て・FK・
記録・VlaSkill の state) は変わらない。`tick` (ロボット側の通し番号) と受信時刻を
snapshot に入れるので、実機の記録から「新しい値がいつ来たか」を後で確かめられる。

`ChannelFactoryInitialize` 済みであることが前提 (actuator の init と同じ)。
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Optional

import numpy as np

from inference.desktop.perception.g1_urdf_fk import G1_JOINT_NAMES
from inference.desktop.perception.joint_state_source import JointStateData

LOWSTATE_TOPIC = "rt/lowstate"
NUM_BODY_JOINTS = len(G1_JOINT_NAMES)  # 29 (lowstate の motor_state は 35 要素)


class LowStateJointSource:
    """`rt/lowstate` の latest-only pull adapter。

    500 Hz で流れてくるので callback は使わず、`get()` で最新 1 件を読む
    (`G1ArmActuator` の lowstate の読み方と同じ)。読めなければ直前の snapshot を返す。

    Args:
        topic: lowstate の DDS topic 名。
        subscriber_factory / message_type: test 用の差し替え口 (既定は SDK)。
    """

    def __init__(
        self,
        topic: str = LOWSTATE_TOPIC,
        *,
        subscriber_factory: Optional[Callable[[str, object], object]] = None,
        message_type: Optional[object] = None,
    ) -> None:
        if subscriber_factory is None or message_type is None:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

            if subscriber_factory is None:
                subscriber_factory = ChannelSubscriber
            if message_type is None:
                message_type = LowState_

        self._latest: Optional[JointStateData] = None
        self._subscriber = subscriber_factory(topic, message_type)
        self._subscriber.Init()

    def get(self) -> Optional[JointStateData]:
        """最新の snapshot を返す (未受信なら None)。"""
        if self._subscriber is None:
            return self._latest
        message = self._subscriber.Read()
        if message is not None:
            snapshot = self._snapshot(message)
            if snapshot is not None:
                self._latest = snapshot
        return self._latest

    def wait_for_first(self, timeout_s: float = 5.0) -> JointStateData:
        """最初の snapshot が来るまで待つ。来なければ RuntimeError。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshot = self.get()
            if snapshot is not None:
                return snapshot
            time.sleep(0.002)
        raise RuntimeError(
            f"timed out waiting for {LOWSTATE_TOPIC} after {timeout_s} s; "
            "joint angles are unavailable"
        )

    @staticmethod
    def _snapshot(message: object) -> Optional[JointStateData]:
        """LowState_ 1 件を JointStateData にする。壊れた sample は None。

        system boundary (SDK からの外部入力) なので、ここだけは広く catch して
        前の snapshot を保つ (`Dex1StateSource._handle` と同じ方針)。
        """
        try:
            motors = message.motor_state  # type: ignore[attr-defined]
            position = np.empty(NUM_BODY_JOINTS, dtype=np.float64)
            velocity = np.empty(NUM_BODY_JOINTS, dtype=np.float64)
            effort = np.empty(NUM_BODY_JOINTS, dtype=np.float64)
            for i in range(NUM_BODY_JOINTS):
                motor = motors[i]
                position[i] = float(motor.q)
                velocity[i] = float(motor.dq)
                effort[i] = float(motor.tau_est)
            received_ns = time.monotonic_ns()
            return JointStateData(
                name=G1_JOINT_NAMES,
                position=position,
                velocity=velocity,
                effort=effort,
                t=received_ns,
                tick=int(getattr(message, "tick", 0)),
                received_monotonic_ns=received_ns,
            )
        except Exception as exc:
            print(f"[LowStateJointSource] invalid lowstate sample: {exc!r}", file=sys.stderr)
            return None

    def close(self) -> None:
        """reader を破棄する。冪等。"""
        subscriber = self._subscriber
        self._subscriber = None
        self._latest = None
        if subscriber is not None:
            subscriber.Close()
