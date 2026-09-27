"""Non-blocking operator gate that actively holds the measured arm pose.

The orchestrator must keep ticking while an operator is deciding whether to start a
learned policy.  Blocking the main thread in ``input()`` would stop boundary-action
publication and camera freshness checks.  This skill reads stdin on a daemon thread
while its normal ``step`` method continues returning the held 14-D arm target.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable, Optional, Sequence

import numpy as np

from inference.desktop.lower_policy.pose_utils import arm_positions_from_joint_state
from inference.desktop.lower_policy.skills.base import Skill


#: 開始姿勢に届いたか。(届いた, 何をどう比べたかの 1 行)。
ArrivalCheck = Callable[[np.ndarray], "tuple[bool, str]"]


def joint_space_arrival(
    measured_arm: Sequence[float],
    target_arm: Sequence[float],
    tolerance_rad: float,
    joint_names: Sequence[str],
) -> tuple[bool, str]:
    """14 関節のうち一番離れた関節で、開始姿勢に届いたかを見る (sdk 経路・joint lane の判定)。"""
    measured = np.asarray(measured_arm, dtype=np.float64).reshape(-1)
    target = np.asarray(target_arm, dtype=np.float64).reshape(-1)
    errors = np.abs(measured - target)
    worst = int(np.argmax(errors))
    return (
        bool(errors[worst] <= tolerance_rad),
        f"worst={joint_names[worst]} error={errors[worst]:.3f}rad "
        f"tolerance={tolerance_rad:.2f}rad",
    )


def discard_pending_stdin() -> None:
    """端末に溜まった入力を捨てる (tty でなければ何もしない)。

    手前の確認 prompt で余分に押された Enter が buffer に残っていると、gate の
    ``input()`` がそれを即座に読み、操作者が姿勢を見る前に policy が始まる。
    """
    try:
        if not sys.stdin.isatty():
            return
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:  # noqa: BLE001 - best effort; input() still gates
        pass


class OperatorConfirmationHoldSkill(Skill):
    """Hold the pose until the operator presses Enter.

    ``input_fn`` is injectable so the confirmation contract can be tested without a
    terminal.  An EOF or input failure is fail-closed and is surfaced through
    ``failure_reason``; it never advances to the policy.

    ``arrival_check`` は問いを出す時点の実測の腕で「開始姿勢に届いたか」を見て、問いの
    文言を変える。手前の準備動作が未到達なら本番経路では先へ進まないが、
    到達判定もこの gate で重ねて確認し、違う姿勢からの開始を防ぐ。
    判定は準備動作と同じもの (pose lane は手先、sdk・joint lane は関節角) を渡す。
    省略時は判定せずに従来の文言。
    """

    #: 入力 thread が自分の問い (操作・到達の詳細) を出すので、orchestrator は
    #: この skill の間は操作表示を上書きしない (上書きすると受け付けるキーが消える)。
    owns_operator_view = True

    def __init__(
        self,
        *,
        name: str,
        next_skill_name: str,
        input_fn: Callable[[str], str] = input,
        discard_pending_input_fn: Callable[[], None] = discard_pending_stdin,
        arrival_check: Optional[ArrivalCheck] = None,
        require_arrival: bool = False,
        hold_arm_target_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self._next_skill_name = str(next_skill_name)
        self._input_fn = input_fn
        self._discard_pending_input_fn = discard_pending_input_fn
        self._arrival_check = arrival_check
        self._require_arrival = bool(require_arrival)
        self._hold_arm_target_provider = hold_arm_target_provider
        self._hold_arm: Optional[np.ndarray] = None
        self._latest_arm: Optional[np.ndarray] = None
        self._confirmed = threading.Event()
        self._reader_started = False
        self._failure_reason: Optional[str] = None
        self._lock = threading.Lock()

    def _on_start(self, params: dict) -> None:
        self._hold_arm = None
        self._latest_arm = None
        self._confirmed.clear()
        self._reader_started = False
        self._failure_reason = None

    def _on_stop(self) -> None:
        pass

    def _prompt(self, arm: np.ndarray) -> str:
        if self._arrival_check is None:
            return (
                f"[gate] {self._next_skill_name} initial arm/hand pose is reached and "
                "actively held. Press Enter to start the policy, or Ctrl+C to stop: "
            )
        reached, detail = self._arrival_check(arm)
        if reached:
            return (
                f"[gate] {self._next_skill_name} initial arm/hand pose is reached "
                f"({detail}) and actively held. Press Enter to start the policy, or "
                "Ctrl+C to stop: "
            )
        instruction = (
            "Enter after arrival, or Ctrl+C to stop: "
            if self._require_arrival
            else "Enter to start the policy from here anyway, or Ctrl+C to stop: "
        )
        return (
            f"[gate] WARNING: {self._next_skill_name} initial arm pose is NOT reached "
            f"({detail}); the arms are held where they are. Check the robot, then press "
            + instruction
        )

    def _read_confirmation(self, prompt: str) -> None:
        try:
            # 姿勢に着く前に押された Enter で policy を始めない。
            self._discard_pending_input_fn()
            while True:
                self._input_fn(prompt)
                with self._lock:
                    latest = None if self._latest_arm is None else self._latest_arm.copy()
                if (
                    not self._require_arrival
                    or self._arrival_check is None
                    or (latest is not None and self._arrival_check(latest)[0])
                ):
                    break
                detail = (
                    "no live arm state"
                    if latest is None else self._arrival_check(latest)[1]
                )
                print(
                    f"[gate] initial pose not reached ({detail}); Enter ignored",
                    file=sys.stderr,
                )
                # 次の問いは今の実測で作り直す (どの関節がどれだけ離れているかを出す)
                prompt = self._prompt(latest if latest is not None else self._hold_arm)
        except (EOFError, OSError) as exc:
            with self._lock:
                self._failure_reason = (
                    f"operator confirmation input failed before "
                    f"{self._next_skill_name}: {exc!r}"
                )
            return
        self._confirmed.set()
        print(
            f"[gate] operator confirmed start of {self._next_skill_name}",
            file=sys.stderr,
        )

    def step(self, obs: dict) -> np.ndarray:
        state = obs.get("joint_state")
        if state is not None:
            latest = arm_positions_from_joint_state(
                tuple(state.name), np.asarray(state.position, dtype=np.float64), self.name
            )
            with self._lock:
                self._latest_arm = latest
        if self._hold_arm is None:
            if state is None:
                raise RuntimeError(f"{self.name} requires a live joint state to hold")
            target = (
                self._hold_arm_target_provider()
                if self._hold_arm_target_provider is not None else None
            )
            self._hold_arm = (
                self._latest_arm.copy() if target is None
                else np.asarray(target, dtype=np.float64).copy()
            )
            if self._hold_arm.shape != (14,) or not np.isfinite(self._hold_arm).all():
                raise RuntimeError(f"{self.name} cannot hold an invalid arm target")
        if not self._reader_started:
            self._reader_started = True
            threading.Thread(
                target=self._read_confirmation,
                args=(self._prompt(self._hold_arm),),
                name=f"{self.name}-stdin",
                daemon=True,
            ).start()
        return self._hold_arm.copy()

    @property
    def is_complete(self) -> bool:
        return self._confirmed.is_set()

    @property
    def failure_reason(self) -> Optional[str]:
        with self._lock:
            return self._failure_reason
