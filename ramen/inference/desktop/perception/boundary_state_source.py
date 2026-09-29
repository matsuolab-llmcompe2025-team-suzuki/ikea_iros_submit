"""Organizer ``:5557`` robot-state stream exposed as ``JointStateData``.

The competition boundary is the sole robot interface at the venue.  In
particular, a boundary run must not quietly keep reading ``rt/lowstate`` just
because that is convenient in the lab: doing so both violates the published
contract and makes the submitted process depend on DDS being configured.

# Dex1 の実測 (`gripper_q`)

CONTRACT は ``body_q`` / ``base_quat`` だけを保証し、手の state は「無い」前提。
一方、運営の state bridge (``reference/orin_bridge/real_orin_state.py``) は同じ
message に **``gripper_q``** (Dex1 の motor の実測 ``q``、``0`` = 閉 /
``-5.30`` = 開) を載せている (2026-09-10 に team の要望で追加)。

legacy経路では無ければ合成 (指令エコー) に戻る。会場のguarded経路は
PC2の読み取り専用guard (:5558) からDDS鮮度確認付きstateだけを受け、合成へ戻らない。
公式のdecodeはそのまま使い、vendorは無改変。
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Any, Optional

import numpy as np

from inference.desktop.perception.g1_urdf_fk import G1_JOINT_NAMES
from inference.desktop.perception.joint_state_source import JointStateData
from inference.desktop.perception.venue_state_guard import MAX_SOURCE_AGE_S

#: 運営 bridge の正規化済み q = -(物理の開き)。raw motorの符号ではない。
#: 我々の model 座標は物理の開き [rad] (0 = 閉) なので符号だけ反転する
#: (`taskspace_adapter.dex1_model_to_taskspace` と同じ前提)。
ORGANIZER_GRIPPER_Q_TO_OPENING = -1.0


def parse_gripper_q(message: Any) -> Optional[np.ndarray]:
    """``g1_debug`` payload の ``gripper_q`` → (left, right) の motor q。

    無い・形が違う・非有限なら None (合成に戻す判断は呼び出し側)。
    """
    if not isinstance(message, dict):
        return None
    gripper = message.get("gripper_q")
    if not isinstance(gripper, dict):
        return None
    values = []
    for side in ("left", "right"):
        entry = gripper.get(side)
        q = entry.get("q") if isinstance(entry, dict) else None
        try:
            value = float(q)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def decode_gripper_aware(blob: bytes):
    """公式 ``StateStream._decode`` (契約 key の形・有限性の検証) の後に ``gripper_q`` も拾う。"""
    from inference.desktop.boundary.states import STATE_TOPIC, StateStream

    state = StateStream._decode(blob)  # 公式の検証 (staticmethod)
    try:
        import msgpack

        prefix = STATE_TOPIC.encode("utf-8")
        message = msgpack.unpackb(blob[len(prefix):], raw=False)
    except Exception:  # noqa: BLE001 - 追加 key は best effort
        message = None
    state.gripper_q = parse_gripper_q(message)
    return state


def gripper_aware_stream_class():
    """公式 ``StateStream`` の decode を通したうえで ``gripper_q`` も拾う class。

    vendor の ``boundary/states.py`` は無改変。``read`` が呼ぶ ``_decode`` だけを
    subclass で包み、公式の検証 (契約 key の形・有限性) を先に通す。
    """
    from inference.desktop.boundary.states import StateStream

    class _GripperAwareStateStream(StateStream):
        def _decode(self, blob: bytes):  # type: ignore[override]
            return decode_gripper_aware(blob)

    return _GripperAwareStateStream


def _gripper_aware_stream(host: str, port: int):
    return gripper_aware_stream_class()(host=host, port=port)


class BoundaryJointStateSource:
    """Latest-only adapter for the organizer's canonical 29-DoF state."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 5557, *, stream: Any = None,
        guarded: bool = False,
    ) -> None:
        self._guarded = guarded
        if stream is None and guarded:
            from inference.desktop.perception.guarded_state_stream import GuardedStateStream
            stream = GuardedStateStream(host, port)
        self._stream = stream if stream is not None else _gripper_aware_stream(host, port)
        self._latest: Optional[JointStateData] = None
        self._latest_gripper: Optional[tuple[np.ndarray, int]] = None
        self._previous_position: Optional[np.ndarray] = None
        self._previous_measured_ns: Optional[int] = None
        self._sample_offset_ns: Optional[int] = None
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._receive_loop,
            name=f"boundary-state:{host}:{port}",
            daemon=True,
        )
        self._thread.start()

    def _receive_loop(self) -> None:
        try:
            self._receive_states()
        finally:
            if getattr(self, "_guarded", False):
                self._stream.close()

    def _receive_states(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
            try:
                state = self._stream.read(timeout_ms=200)
            except Exception:
                # Invalid wire messages are not converted into a fresh sample.
                # The orchestrator's joint-state freshness guard stops the run.
                continue
            if state is None:
                continue

            position = np.asarray(state.body_q, dtype=np.float64).copy()
            base_quat = np.asarray(state.base_quat, dtype=np.float64).copy()
            now_ns = time.monotonic_ns()
            # received_ns: freshness. measured_ns: sample spacing (velocity, dwell).
            received_ns = measured_ns = now_ns
            if getattr(self, "_guarded", False):
                received_ns = getattr(state, "source_received_ns", None)
                sample_ns = getattr(state, "source_sample_ns", None)
                if (
                    received_ns is None or sample_ns is None
                    or not 0 <= now_ns - received_ns <= MAX_SOURCE_AGE_S * 1e9
                ):
                    continue
                # The freshness stamp is a lower bound (request time - source age), so a
                # reply delayed in transit is early by that delay: its spacing is not the
                # sample spacing. Measure on the guard's sample clock (one PC2 monotonic
                # clock), anchored once to ours; only differences are used.
                if self._sample_offset_ns is None:
                    self._sample_offset_ns = received_ns - sample_ns
                measured_ns = sample_ns + self._sample_offset_ns
                if self._previous_measured_ns is not None and measured_ns <= self._previous_measured_ns:
                    continue
            velocity = np.zeros_like(position)
            if self._previous_position is not None and self._previous_measured_ns is not None:
                dt = (measured_ns - self._previous_measured_ns) * 1e-9
                if 1e-4 <= dt <= 1.0:
                    velocity = (position - self._previous_position) / dt
            self._previous_position = position
            self._previous_measured_ns = measured_ns
            snapshot = JointStateData(
                name=G1_JOINT_NAMES,
                position=position,
                velocity=velocity,
                effort=np.zeros_like(position),
                t=received_ns,
                tick=getattr(state, "source_tick", None),
                received_monotonic_ns=received_ns,
                base_quat_wxyz=base_quat,
                measured_monotonic_ns=measured_ns,
            )
            gripper = getattr(state, "gripper_q", None)
            with self._lock:
                if self._closed:
                    return
                self._latest = snapshot
                if gripper is not None:
                    self._latest_gripper = (np.asarray(gripper).copy(), received_ns)

    def get(self) -> Optional[JointStateData]:
        with self._lock:
            return None if self._closed else self._latest

    def latest_gripper(self) -> Optional[tuple[np.ndarray, int]]:
        """最後に受けた ``gripper_q`` (motor q, 受信 monotonic ns)。一度も無ければ None。"""
        with self._lock:
            if self._closed or self._latest_gripper is None:
                return None
            q, received = self._latest_gripper
            return q.copy(), received

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            stream = self._stream
            thread = self._thread
        # A ZMQ socket must not be closed by this thread while the receiver
        # thread is inside poll()/recv().  ``read`` has a 200 ms timeout, so
        # first let that call return and observe ``_closed``, then close the
        # socket from its owner lifecycle.  This mirrors ZmqFrameSource and
        # avoids libzmq assertion failures during normal/abnormal shutdown.
        if thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if thread.is_alive():
            raise RuntimeError("boundary state receiver did not stop within 2s")
        stream.close()


class BoundaryDex1StateSource:
    """公式 ``:5557`` の ``gripper_q`` があれば実測、無ければ指令エコーの合成。

    ``gripper_q`` を一度でも受けたら以後は実測を返す (受信時刻つき。途絶えれば
    古い値として orchestrator の鮮度検査が止める)。合成へ黙って戻らない。
    受ける前は ``synthetic`` (``SyntheticDex1StateSource``) をそのまま返す
    (``measured=False``)。
    """

    def __init__(self, joint_source: BoundaryJointStateSource, synthetic: Any) -> None:
        self._joint = joint_source
        self._synthetic = synthetic
        self._measured_seen = False
        self._reported = False
        self._lock = threading.Lock()

    def bind_command_source(self, command_source: object) -> None:
        self._synthetic.bind_command_source(command_source)

    @property
    def measured_available(self) -> bool:
        with self._lock:
            return self._measured_seen

    def get(self):
        from inference.desktop.perception.dex1_state_source import Dex1StateData

        sample = self._joint.latest_gripper()
        with self._lock:
            first = sample is not None and not self._measured_seen
            if sample is not None:
                self._measured_seen = True
            report = not self._reported
            self._reported = True
        if first:
            print(
                "[dex1] official :5557 carries gripper_q; using MEASURED Dex1 "
                "state (opening = -q)",
                file=sys.stderr,
            )
        elif report and sample is None:
            if getattr(self._joint, "_guarded", False):
                print("[dex1] waiting for guarded measured state; synthetic fallback disabled", file=sys.stderr)
            else:
                print(
                    "[dex1] no gripper_q on :5557 yet; Dex1 state is the SYNTHETIC "
                    "command echo (grasp checks fall back to VLM-only)",
                    file=sys.stderr,
                )
        if sample is None:
            if getattr(self._joint, "_guarded", False):
                return None
            return self._synthetic.get()
        q, received_ns = sample
        return Dex1StateData(
            position_rad=ORGANIZER_GRIPPER_Q_TO_OPENING * q,
            left_received_monotonic_ns=received_ns,
            right_received_monotonic_ns=received_ns,
            t=time.time_ns(),
            measured=True,
        )

    def close(self) -> None:
        self._synthetic.close()
