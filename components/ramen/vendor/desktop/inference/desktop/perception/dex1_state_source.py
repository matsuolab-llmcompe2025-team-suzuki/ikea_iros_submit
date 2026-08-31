"""Latest-only read-only source for the two Dex1-1 motor states.

The official ``dex1_1_gripper_server`` publishes the motor output position in
physical radians on ``rt/dex1/{left,right}/state``.  It does not normalize the
value to ``[0, 1]``.  This adapter deliberately preserves those units because
the Team RAMEN LeRobot datasets store ``hand_state`` and ``hand_cmd`` in the
same physical coordinate (typically 0--4.5 rad).

``ChannelFactoryInitialize`` must be called before constructing this class.
The source only creates DDS subscribers; it cannot command either gripper.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


DEX1_LEFT_STATE_TOPIC = "rt/dex1/left/state"
DEX1_RIGHT_STATE_TOPIC = "rt/dex1/right/state"


@dataclass(frozen=True)
class Dex1StateData:
    """Atomic bilateral Dex1 snapshot in physical motor-output radians."""

    position_rad: np.ndarray
    left_received_monotonic_ns: int
    right_received_monotonic_ns: int
    t: int


class Dex1StateSource:
    """Subscribe to both official Dex1 state topics and retain the latest pair."""

    def __init__(
        self,
        left_topic: str = DEX1_LEFT_STATE_TOPIC,
        right_topic: str = DEX1_RIGHT_STATE_TOPIC,
        *,
        subscriber_factory: Optional[Callable[[str, object], object]] = None,
        message_type: Optional[object] = None,
    ) -> None:
        if subscriber_factory is None or message_type is None:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

            if subscriber_factory is None:
                subscriber_factory = ChannelSubscriber
            if message_type is None:
                message_type = MotorStates_

        self._lock = threading.Lock()
        self._closed = False
        self._positions: list[float | None] = [None, None]
        self._received_ns = [0, 0]
        self._latest: Dex1StateData | None = None

        self._left_subscriber = subscriber_factory(left_topic, message_type)
        self._right_subscriber = subscriber_factory(right_topic, message_type)
        self._left_subscriber.Init(self._handle_left, 1)
        self._right_subscriber.Init(self._handle_right, 1)

    def _handle_left(self, message: object) -> None:
        self._handle(0, message)

    def _handle_right(self, message: object) -> None:
        self._handle(1, message)

    def _handle(self, side: int, message: object) -> None:
        try:
            states = getattr(message, "states", None)
            if not states:
                raise ValueError("empty MotorStates.states")
            q = float(states[0].q)
            if not math.isfinite(q):
                raise ValueError(f"non-finite motor position: {q!r}")
            received_ns = time.monotonic_ns()
            with self._lock:
                if self._closed:
                    return
                self._positions[side] = q
                self._received_ns[side] = received_ns
                if self._positions[0] is None or self._positions[1] is None:
                    return
                positions = np.asarray(self._positions, dtype=np.float32)
                self._latest = Dex1StateData(
                    position_rad=positions,
                    left_received_monotonic_ns=self._received_ns[0],
                    right_received_monotonic_ns=self._received_ns[1],
                    t=max(self._received_ns),
                )
        except Exception as exc:
            side_name = "left" if side == 0 else "right"
            print(
                f"[Dex1StateSource] invalid {side_name} state: {exc!r}",
                file=sys.stderr,
            )

    def get(self) -> Dex1StateData | None:
        """Return the latest atomic bilateral sample, or ``None`` until ready."""

        with self._lock:
            return self._latest

    def close(self) -> None:
        """Close both readers.  Idempotent and never creates a command writer."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            left = self._left_subscriber
            right = self._right_subscriber
            self._left_subscriber = None
            self._right_subscriber = None
            self._latest = None
        for subscriber in (left, right):
            if subscriber is not None:
                subscriber.Close()
