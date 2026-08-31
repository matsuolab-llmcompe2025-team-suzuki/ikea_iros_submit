"""Orchestrator の frame 入口を抽象化する FrameSource protocol と実装。

Orchestrator は `source.get()` で 1 frame ずつ pull する形。source の実装差
(LeRobot ep replay / ROS2 real-time) は adapter に閉じる。

- LerobotFrameSource: sequence source (Iterable → 次 frame)。ep 終端で None を返す。
- Ros2FrameSource: ROS2 CompressedImage topic を cyclonedds 直接で subscribe
  (rclpy を使わない、Issue #58 参照)。SDK の ChannelFactory 経由で subscribe を
  register し、callback で受け取った最新 1 frame のみ保持する latest-only policy。
  cv2 / cyclonedds / SDK / IDL は lazy import なので main env (cv2/cyclonedds 未
  install) から本 module を import しても壊れない。topology β (Desktop 頭脳 / Orin I/O) の
  primary source (Orin が camera を publish、Desktop が subscribe)。
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class FrameData:
    """Orchestrator の tick 単位。

    Attributes:
        rgb: HWC BGR (cv2 native)。YOLO predict の入力形式に一致。
        t: timestamp。LeRobot ep replay では frame_index (int)、ROS2 real-time では
            header.stamp の nanosecond。Orchestrator の重複 tick 防止 (t == last_t)
            にのみ使う。単調性の保証は source 側 (Lerobot は本質的に単調、ROS2 は
            wall clock)。
    """

    rgb: np.ndarray
    t: int
    # Host monotonic time when the DDS callback received the JPEG. ROS header
    # stamps from
    # independent USB cameras can use unrelated device-clock epochs and must
    # not be subtracted across streams.  Optional keeps replay/test callers
    # source-compatible.
    received_monotonic_ns: int | None = None


class FrameSource(Protocol):
    """Orchestrator が pull する frame 入口の abstract。

    LeRobot ep replay と ROS2 real-time が同じ interface で振る舞う。
    """

    def get(self) -> Optional[FrameData]:
        """最新 1 frame を返す。無ければ None (stream 終了 / 未受信)。"""
        ...


class LerobotFrameSource:
    """LeRobot ep replay からの pull source (sequence の次 frame を返す adapter)。

    具体的な LeRobot dataset 読みは caller (e2e script) に任せる。ここは
    「Iterable → get()」の変換だけ = framework-agnostic、testable。
    ep 終端 (iter 尽きた後) は永久に None を返す。

    Args:
        frames: FrameData の Iterable。caller が LeRobotDataset から 1 ep 分を
            展開して渡す想定。
    """

    def __init__(self, frames: Iterable[FrameData]) -> None:
        self._iter: Iterator[FrameData] = iter(frames)
        self._exhausted: bool = False

    def get(self) -> Optional[FrameData]:
        if self._exhausted:
            return None
        try:
            return next(self._iter)
        except StopIteration:
            self._exhausted = True
            return None


TRAINING_WRIST_HEIGHT: int = 480
TRAINING_WRIST_WIDTH: int = 640


def adapt_wrist_to_training_shape(frame_bgr: np.ndarray) -> np.ndarray:
    """Center-crop the D405 wrist frame to the training input aspect (640x480).

    実機 D405 は 848x480 native で強制 (Orin driver が 640 指定を silently 848 に
    fallback する、docs/inference/g1_hardware_operational_notes.md §4)。訓練
    データ (BitRobot 提供) は wrist が 480x640 に統一されているため、そのまま
    policy に流すと 224x224 resize 時の x 圧縮率が訓練と違い、モデルが OOD 画像
    を見る。ここで中央 640 列を切り出して aspect を揃える。

    Args:
        frame_bgr: (480, 848, 3) or (480, 640, 3) uint8 BGR。

    Returns:
        (480, 640, 3) uint8 BGR。
    """
    if frame_bgr.shape == (TRAINING_WRIST_HEIGHT, TRAINING_WRIST_WIDTH, 3):
        return frame_bgr
    if frame_bgr.shape == (TRAINING_WRIST_HEIGHT, 848, 3):
        excess = (848 - TRAINING_WRIST_WIDTH) // 2  # 104
        return np.ascontiguousarray(
            frame_bgr[:, excess : excess + TRAINING_WRIST_WIDTH, :]
        )
    raise ValueError(
        "wrist frame must be (480, 640, 3) or (480, 848, 3) uint8 BGR, got "
        f"shape={frame_bgr.shape} dtype={frame_bgr.dtype}"
    )


class Ros2FrameSource:
    """ROS2 CompressedImage (JPEG) topic からの latest-only pull adapter。

    rclpy を使わず cyclonedds を直接使って subscribe する (Issue #58 の Scope 決定
    経緯参照)。SDK 側の `ChannelFactory` singleton (`G1SDKWalkActuator.__init__`
    が既に `ChannelFactoryInitialize` を呼んで確保している) を流用して subscribe を
    register する = actuator init 後に本 class を instantiate する順序依存がある。

    ROS2 は callback-driven (push)、Orchestrator は tick pull を要求するため、
    subscription callback で受け取った最新 1 frame だけ保持し get() で返す。
    Orchestrator が tick で遅れても buffer は詰まらず (常に最新を上書き)、
    frame drop で吸収する = skill 判定 layer では許容可能。

    DDS callbackではJPEG bytesと時刻をlatest-onlyで保存するだけにする。decodeを
    callback内で行うと複数カメラのDDS readerがPython decode待ちで互いを阻害する
    ため、`get()`が新しいgenerationを初めて読む時だけdecodeする。

    cv2 / cyclonedds / SDK / IDL は `__init__` 内で lazy import する。main env
    (cv2 / cyclonedds 未 install) から本 module 全体を import しても、Ros2FrameSource
    を instantiate しなければ ImportError にならない (LerobotFrameSource など他 class
    は正常に使える)。

    Args:
        topic: CompressedImage topic 名 (例: /head/camera/color/image_raw/compressed)。
        stereo_view: packed stereo image の扱い。``"packed"`` は全幅、``"left"`` /
            ``"right"`` は左右半分を返す。YOLO / policy が LeRobot の head_left
            で学習されている場合は ``"left"`` を指定する。
        qos: cyclonedds `Qos` object。None なら SensorDataQoS 相当を default で構築
            (Reliability=BestEffort, History=KeepLast(1), Durability=Volatile)。
            camera image は sensor data なので drop 許容 = BestEffort が適切。
    """

    def __init__(
        self,
        topic: str,
        qos: Optional[object] = None,
        *,
        stereo_view: str = "packed",
    ) -> None:
        if stereo_view not in {"packed", "left", "right"}:
            raise ValueError(
                "stereo_view must be one of 'packed', 'left', 'right', "
                f"got {stereo_view!r}"
            )
        self._stereo_view = stereo_view

        # cv2 / cyclonedds / SDK / IDL は runtime env にしか無いので lazy import。
        # cv2 は listener thread の _cb で使うため instance に保持する。
        import cv2

        from cyclonedds.core import Policy
        from cyclonedds.qos import Qos
        from unitree_sdk2py.core.channel import ChannelFactory

        from inference.desktop.perception.sensor_msgs_idl import CompressedImage_

        self._cv2 = cv2

        if qos is None:
            # SensorDataQoS (ROS2 の rclpy.qos.qos_profile_sensor_data 等価)
            qos = Qos(
                Policy.Reliability.BestEffort,
                Policy.History.KeepLast(1),
                Policy.Durability.Volatile,
            )

        self._latest: Optional[FrameData] = None
        self._pending: tuple[bytes, int, int] | None = None
        self._received_count = 0
        self._decoded_count = 0
        self._last_jpeg_size = 0
        self._lock = threading.Lock()
        self._closed = False
        self._decode_event = threading.Event()
        self._decode_thread: threading.Thread | None = threading.Thread(
            target=self._decoder_loop,
            name=f"camera-decode:{topic}",
            daemon=True,
        )
        self._decode_thread.start()
        # SDK ChannelFactory singleton (actuator が先に Init 済み前提) から
        # channel を作り、SetReader で subscribe register (handler=self._cb)。
        # queueLen=0 は listener thread から直接 handler を呼ぶ意 (queue 挟まない)。
        # rmw_cyclonedds maps a ROS topic ``/foo`` to the DDS topic ``rt/foo``.
        # unitree_sdk2py uses raw DDS topic names, so apply the ROS 2 mapping
        # explicitly while still accepting an already-mapped DDS name.
        dds_topic = f"rt{topic}" if topic.startswith("/") else topic
        self._channel = ChannelFactory().CreateChannel(dds_topic, CompressedImage_)
        self._channel.SetReader(qos=qos, handler=self._cb, queueLen=0)

    def _cb(self, msg: object) -> None:
        """cyclonedds listener thread から発火。JPEG bytesをlatest更新。

        System boundary (ROS2 msg = external input) なので防御的に:
          - broad exception catch → cyclonedds listener thread が伝播死しないよう
            保護 (msg 構造 mismatch / numpy 変換 error 等で subscription が silent
            停止するのを防ぐ)

        `msg` の shape (cyclonedds が deserialize した `CompressedImage_` instance):
          - `msg.data`: sequence[uint8] (bytes or list of int、実装差吸収のため
            bytes() で正規化してから np.frombuffer に渡す)
          - `msg.header.stamp.sec`: int32
          - `msg.header.stamp.nanosec`: uint32
        """
        try:
            data_bytes = bytes(msg.data)  # type: ignore[attr-defined]
            stamp = msg.header.stamp  # type: ignore[attr-defined]
            t_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            received_ns = time.monotonic_ns()
            with self._lock:
                # DDS listener callback と shutdown が競合しても、close 後に
                # stale frame を再登録しない。
                if self._closed:
                    return
                self._pending = (data_bytes, t_ns, received_ns)
                self._received_count += 1
                self._last_jpeg_size = len(data_bytes)
            self._decode_event.set()
        except Exception as e:
            print(f"[Ros2FrameSource] _cb error: {e!r}", file=sys.stderr)

    def _decode_latest_pending(self) -> Optional[FrameData]:
        with self._lock:
            if self._closed:
                return None
            pending = self._pending
            latest = self._latest
        if pending is None:
            return latest
        data_bytes, t_ns, received_ns = pending
        if latest is not None and latest.t == t_ns:
            return latest

        try:
            rgb = self._cv2.imdecode(
                np.frombuffer(data_bytes, np.uint8),
                self._cv2.IMREAD_COLOR,
            )
            if rgb is None:
                return latest
            if self._stereo_view != "packed":
                width = rgb.shape[1]
                if width < 2 or width % 2 != 0:
                    raise ValueError(
                        "packed stereo image must have a positive even width, "
                        f"got {width}"
                    )
                split = width // 2
                rgb = (
                    rgb[:, :split].copy()
                    if self._stereo_view == "left"
                    else rgb[:, split:].copy()
                )
            decoded = FrameData(
                rgb=rgb,
                t=t_ns,
                received_monotonic_ns=received_ns,
            )
        except Exception as e:
            print(f"[Ros2FrameSource] decode error: {e!r}", file=sys.stderr)
            return latest

        with self._lock:
            if self._closed:
                return None
            # A newer pending JPEG may have arrived while decoding.  Returning
            # this valid frame once is safe; the next get decodes the newer one.
            self._latest = decoded
            self._decoded_count += 1
            return decoded

    def _decoder_loop(self) -> None:
        """Decode each camera independently, retaining only its newest JPEG."""
        while True:
            self._decode_event.wait(timeout=0.1)
            self._decode_event.clear()
            with self._lock:
                if self._closed:
                    return
            while True:
                with self._lock:
                    pending_before = self._pending
                self._decode_latest_pending()
                with self._lock:
                    if self._closed:
                        return
                    if self._pending is pending_before:
                        break

    def get(self) -> Optional[FrameData]:
        """Return the latest frame without decoding on the caller thread."""
        # Unit tests construct the object without __init__; preserve a small
        # synchronous seam there.  Production always owns a decoder worker.
        if getattr(self, "_decode_thread", None) is None:
            return self._decode_latest_pending()
        with self._lock:
            return None if self._closed else self._latest

    def diagnostics(self) -> dict[str, int]:
        """Return monotonic counters for non-actuating camera smoke checks."""
        with self._lock:
            return {
                "received_count": self._received_count,
                "decoded_count": self._decoded_count,
                "last_jpeg_size": self._last_jpeg_size,
            }

    def close(self) -> None:
        """DDS camera reader を明示的に破棄する。冪等。

        CycloneDDS の listener / native receive thread を Python interpreter
        shutdown より前に停止する必要がある。これを省略すると、listener callback
        を保持したまま interpreter teardown へ入り、native ``recvMC`` thread が
        abort / segfault することがある。
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            channel = self._channel
            self._channel = None
            self._pending = None
            self._latest = None
            decode_event = self._decode_event
            decode_thread = self._decode_thread
            self._decode_thread = None

        decode_event.set()
        if decode_thread is not None:
            decode_thread.join(timeout=2.0)
            if decode_thread.is_alive():
                print(
                    "[Ros2FrameSource] decode worker did not stop within 2s",
                    file=sys.stderr,
                )

        # SDK の Channel.CloseReader() が cyclonedds DataReader を破棄する。
        # lock 外で実行し、listener teardown が callback を待つ場合の deadlock を
        # 避ける。
        if channel is not None:
            channel.CloseReader()
