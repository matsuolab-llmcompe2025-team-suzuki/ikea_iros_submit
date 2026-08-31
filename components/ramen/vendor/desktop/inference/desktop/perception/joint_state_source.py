"""ROS2 sensor_msgs/JointState を cyclonedds direct で subscribe する source (Issue #75)。

Orin 側 `real_hw_bridge_node` (Issue #65) が publish する `/joint_states` を
Desktop 側で受けて orchestrator obs に joint 位置 / 速度 / トルク推定を流す。
Ros2FrameSource と完全に同じ pattern (cyclonedds direct、latest-only、lazy import)。

topology β 前提 (Desktop 頭脳 / Orin I/O): Orin が real_hw_bridge_node で
`sensor_msgs.msg.JointState` を rclpy publish、Desktop 側はこの source が
cyclonedds ChannelSubscriber で subscribe (rclpy 使わず、Issue #58 決定)。
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class JointStateData:
    """Orchestrator obs に渡す joint state snapshot。

    Attributes:
        name: joint 名の順序 (G1 = 29 entries、joint_mapping.G1_JOINT_NAMES と同順)。
            VLA model 側で index → 意味 mapping に使う。
        position: 関節位置 [rad]、shape (N,) float64 ndarray。
        velocity: 関節速度 [rad/s]、shape (N,) float64 ndarray。
        effort: 関節トルク推定 [Nm]、shape (N,) float64 ndarray。
        t: `header.stamp` の nanosecond。orchestrator の tick sync 材料。
    """

    name: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    t: int


class JointStateSource:
    """`/joint_states` topic からの latest-only pull adapter。

    rclpy を使わず cyclonedds を直接使って subscribe する (Issue #58 の Scope 決定
    経緯参照)。SDK 側の `ChannelFactory` singleton (actuator init で確保済み前提)
    を流用する = actuator init 後に本 class を instantiate する順序依存がある
    (Ros2FrameSource と同じ)。

    Callback で受け取った最新 1 msg だけ保持 → get() で latest snapshot を返す。
    orchestrator tick が遅れても buffer は詰まらず (常に最新を上書き)、
    joint state は 50Hz publish で obs の tick rate (30Hz) を上回るので drop 許容。

    cyclonedds / SDK / IDL は `__init__` 内で lazy import する。main env
    (cyclonedds 未 install) から本 module 全体を import しても、
    JointStateSource を instantiate しなければ ImportError にならない。

    Args:
        topic: JointState topic 名 (default `/joint_states`、Issue #65 で
            real_hw_bridge_node が publish する topic 名と一致)。
        qos: cyclonedds `Qos` object。None なら SensorDataQoS 相当を default で構築
            (Reliability=BestEffort, History=KeepLast(1), Durability=Volatile)。
            joint state は sensor data なので drop 許容 = BestEffort が適切。
    """

    def __init__(
        self,
        topic: str = "/joint_states",
        qos: Optional[object] = None,
    ) -> None:
        from cyclonedds.core import Policy
        from cyclonedds.qos import Qos
        from unitree_sdk2py.core.channel import ChannelFactory

        from inference.desktop.perception.sensor_msgs_idl import JointState_

        if qos is None:
            qos = Qos(
                Policy.Reliability.BestEffort,
                Policy.History.KeepLast(1),
                Policy.Durability.Volatile,
            )

        self._latest: Optional[JointStateData] = None
        self._lock = threading.Lock()
        self._closed = False
        # rmw_cyclonedds maps a ROS topic `/foo` to the DDS topic `rt/foo`。
        # Ros2FrameSource と同じ pattern (raw DDS name も一応通せるよう分岐)。
        dds_topic = f"rt{topic}" if topic.startswith("/") else topic
        self._channel = ChannelFactory().CreateChannel(dds_topic, JointState_)
        self._channel.SetReader(qos=qos, handler=self._cb, queueLen=0)

    def _cb(self, msg: object) -> None:
        """cyclonedds listener thread から発火。latest snapshot を更新。

        System boundary (ROS2 msg = external input) なので防御的に:
          - broad exception catch → cyclonedds listener thread が伝播死しないよう
            保護 (msg 構造 mismatch / numpy 変換 error 等で subscription が silent
            停止するのを防ぐ、Ros2FrameSource._cb と同 pattern)。
        """
        try:
            name = tuple(str(n) for n in msg.name)  # type: ignore[attr-defined]
            position = np.asarray(msg.position, dtype=np.float64)  # type: ignore[attr-defined]
            velocity = np.asarray(msg.velocity, dtype=np.float64)  # type: ignore[attr-defined]
            effort = np.asarray(msg.effort, dtype=np.float64)  # type: ignore[attr-defined]
            stamp = msg.header.stamp  # type: ignore[attr-defined]
            t_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            snapshot = JointStateData(
                name=name,
                position=position,
                velocity=velocity,
                effort=effort,
                t=t_ns,
            )
            with self._lock:
                # close 後は stale snapshot を再登録しない (native teardown 競合対策)。
                if self._closed:
                    return
                self._latest = snapshot
        except Exception as e:
            print(f"[JointStateSource] _cb error: {e!r}", file=sys.stderr)

    def get(self) -> Optional[JointStateData]:
        """最新 snapshot を返す (未受信 or before-first-msg は None)。"""
        with self._lock:
            return self._latest

    def close(self) -> None:
        """DDS reader を明示的に破棄する。冪等 (Ros2FrameSource.close と同型、#101)。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            channel = self._channel
            self._channel = None
            self._latest = None
        if channel is not None:
            channel.CloseReader()
