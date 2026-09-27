"""Dex1 を目標の開度へ動かす skill (Issue #141 束 1-8 / D2)。

# なぜ要るか

評価経路は skill を始める前に「手を全開にする → 腕を開始姿勢へ → 手を教師の frame 0 の
開度へ」を必ず通っていたのに、本番経路は手を前の状態のまま policy を始めていた
(INF-14)。学習データの frame 0 は必ず決まった開度から始まるので、開始の手の開度が
違うと最初の chunk から入力が学習と違う。

その手順を「腕を保持したまま手だけ動かす」skill にして、頭の手順
(手を開く → 腕の pre-motion → 手を開始の開度へ → N 秒保持) の 1 つとして
評価でも本番でも同じものを使う。

# 3 つの目標

| target | 目標の開度 | 接触で止まってよいか |
|---|---|---|
| `open` | 物理全開 (`open_rad`: SDK 5.4 rad / boundary 5.3 rad) | いいえ |
| `pose` | その skill の教師 frame 0 の開度 (`initial_pose.dex1_opening_fraction` × 4.5) | いいえ |
| `grasp` | 同上 | **はい** (脚を掴んで止まる。力センサが無いので「動かなくなった位置」で判定) |
| `release` | 握っている手だけ `open_rad`。他の手は直前の指令のまま | いいえ |

数値は `skill_config.yaml` の `hand_pre_motion` section (評価経路の実績値)。

# 既に物を握っている手 (model の境界、Issue #159 B4a-04)

pick -> insert の境界では、pick の終わりに脚を握った手へ `grasp` を掛ける。

- ramp は **直前の指令** から始める (`read_last_commanded_positions`)。実測から
  始めると、脚で止まっている実測 (2.2 前後) まで指令が緩み、把持力が一瞬抜ける
  (Dex1 は P 制御で、力は「実測 - 指令」に比例する)。読めない actuator では実測から。
- 実測が直前の指令より `holding_contact_gap_rad` 以上開いていて、目標が閉じる向きの
  側は「握っている」とみなす。1 mm も閉じられないので、80% 進む条件と「動かない」
  watchdog を外し、安定したら `holding_contact` で完了にする。
- 力センサが無いので、握って止まった手と固まった手は区別できない。前の model が
  握って終わった直後なので、握っていると解釈する。

# 腕を上げる前に握っている手を離す (`release`、Issue #159 T3)

pick -> insert 以外の境界の先頭で使う。握っている = 実測が直前の指令より
`holding_contact_gap_rad` 以上開いていて、実測が `holding_min_opening_rad` 以上
(握り拳は物を持てない)。直前の指令が読めない actuator では、実測が
`holding_min_opening_rad` 以上の手を全部離す。離す手が無ければ指令を 1 つも出さずに
その tick で完了する (教師どおりに離して終わっていれば時間を使わない)。
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Callable, Optional

import numpy as np

from inference.desktop.lower_policy.initial_pose import initial_pose_from_config
from inference.desktop.lower_policy.pose_utils import arm_positions_from_joint_state
from inference.desktop.lower_policy.skills.base import Skill
from inference.desktop.lower_policy.skills.hand_ramp import (
    STABLE_HISTORY_LEN,
    build_hand_target_ramp,
    hand_target_completion_mode,
    resolve_hand_opening_rad,
)

NO_MOTION_WATCHDOG_SEC = 2.0
NO_MOTION_MIN_PROGRESS_RAD = 0.03
# 1 tick に数える制御時間の上限 [s]。腕の pre-motion (#152 97e502d) と同じ。
# tick が止まった間 (model の読み込みで GIL を取られる / gc) は数えない。
MAX_TICK_CONTROL_SEC = 0.1

DEX1_FAULT_DESCRIPTIONS = {
    0x001: "overcurrent",
    0x002: "transient overvoltage",
    0x004: "continuous overvoltage",
    0x008: "transient undervoltage",
    0x010: "controller overheating",
    0x020: "MOS temperature fault",
    0x040: "MOS temperature sensor fault",
    0x080: "housing overheating",
    0x100: "housing temperature sensor fault",
    0x200: "winding overheating",
    0x400: "rotor encoder 1 fault",
    0x800: "rotor encoder 2 fault",
    0x1000: "output encoder fault",
}

TARGET_MODES = ("open", "pose", "grasp", "release")


def _measured_arm_14(obs: dict) -> np.ndarray:
    """obs の joint_state から腕 14 関節を取る (名前で引く)。"""
    state = obs.get("joint_state")
    if state is None:
        raise RuntimeError("hand pre-motion requires a live joint state to hold the arms")
    return arm_positions_from_joint_state(
        tuple(state.name), np.asarray(state.position, dtype=np.float64), "hand pre-motion"
    )


class HandPreMotionSkill(Skill):
    """腕を保持したまま Dex1 を目標の開度へ動かす。

    Args:
        name: skill 名 (dispatcher の key)。
        hand_actuator: `send_action([left, right])` を持つもの。
        target_rad: 目標の開度 (left, right) [rad]。
        allow_closing_contact: 閉じる途中で接触して安定したら完了にしてよいか。
        velocity_limit_rad_s / tolerance_rad / required_stable_samples: ramp と到達の判定。
        minimum_timeout_sec / timeout_margin_sec: 締め切り = max(minimum, 所要時間 + margin)。
        state_max_age_s: これより古い Dex1 の state は使わない。
        control_hz: ramp を刻む周期 (tick の周期)。
    """

    def __init__(
        self,
        name: str,
        *,
        hand_actuator: Any,
        target_rad: tuple[float, float],
        allow_closing_contact: bool = False,
        velocity_limit_rad_s: float = 1.5,
        tolerance_rad: float = 0.05,
        required_stable_samples: int = 5,
        minimum_timeout_sec: float = 5.0,
        timeout_margin_sec: float = 3.0,
        state_max_age_s: float = 0.5,
        control_hz: float = 30.0,
        time_fn: Callable[[], float] = time.monotonic,
        holding_contact_gap_rad: float = 0.10,
        release: bool = False,
        holding_min_opening_rad: float = 1.0,
        hold_arm_target_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> None:
        super().__init__()
        target = np.asarray(target_rad, dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError(f"target_rad must be finite 2-D, got {target_rad!r}")
        for label, value in (
            ("velocity_limit_rad_s", velocity_limit_rad_s),
            ("tolerance_rad", tolerance_rad),
            ("minimum_timeout_sec", minimum_timeout_sec),
            ("state_max_age_s", state_max_age_s),
            ("control_hz", control_hz),
            ("holding_contact_gap_rad", holding_contact_gap_rad),
            ("holding_min_opening_rad", holding_min_opening_rad),
        ):
            if value <= 0.0:
                raise ValueError(f"{label} must be > 0, got {value}")
        if required_stable_samples <= 0:
            raise ValueError(
                f"required_stable_samples must be > 0, got {required_stable_samples}"
            )
        self.name = name
        self._hand_actuator = hand_actuator
        self._target = target
        self._allow_closing_contact = bool(allow_closing_contact)
        self._velocity_limit_rad_s = float(velocity_limit_rad_s)
        self._tolerance_rad = float(tolerance_rad)
        self._required_stable_samples = int(required_stable_samples)
        self._minimum_timeout_sec = float(minimum_timeout_sec)
        self._timeout_margin_sec = float(timeout_margin_sec)
        self._state_max_age_s = float(state_max_age_s)
        self._control_hz = float(control_hz)
        self._time_fn = time_fn
        self._holding_contact_gap_rad = float(holding_contact_gap_rad)
        self._release = bool(release)
        self._holding_min_opening_rad = float(holding_min_opening_rad)
        self._hold_arm_target_provider = hold_arm_target_provider
        if self._release and self._allow_closing_contact:
            raise ValueError("release never closes, so it cannot accept closing contact")
        # release は _begin_ramp で手ごとの目標を決め直すので、設定値を残しておく
        self._configured_target = target.copy()
        self._reset()

    @classmethod
    def from_config(
        cls,
        skill_config: dict,
        skill_name: str,
        *,
        target: str,
        hand_actuator: Any,
        name: Optional[str] = None,
        **overrides: Any,
    ) -> "HandPreMotionSkill":
        """`skill_config.yaml` から作る。

        Args:
            skill_config: `yaml.safe_load(skill_config.yaml)` の結果。
            skill_name: 開度の出どころになる学習 skill (`skills.<name>.initial_pose`)。
            target: `open` / `pose` / `grasp`。
            name: skill 名 (既定は `hand_<target>_<skill_name>`)。
        """
        if target not in TARGET_MODES:
            raise ValueError(f"target must be one of {TARGET_MODES}, got {target!r}")
        settings = dict(skill_config.get("hand_pre_motion") or {})
        unknown = sorted(
            set(settings)
            - {
                "velocity_limit_rad_s", "tolerance_rad", "required_stable_samples",
                "minimum_timeout_sec", "timeout_margin_sec", "state_max_age_s",
                "holding_contact_gap_rad", "holding_min_opening_rad",
                "open_rad",
            }
        )
        if unknown:
            raise ValueError(f"hand_pre_motion has unknown keys: {unknown}")
        # 開く・離すは同じ開度 (Issue #159 T8)
        open_rad = resolve_hand_opening_rad(skill_config)
        settings.pop("open_rad", None)
        if target in ("open", "release"):
            target_rad = (open_rad, open_rad)
        else:
            target_rad = initial_pose_from_config(skill_config, skill_name).dex1_target_rad
        settings.update(overrides)
        return cls(
            name or f"hand_{target}_{skill_name}",
            hand_actuator=hand_actuator,
            target_rad=target_rad,
            allow_closing_contact=(target == "grasp"),
            release=(target == "release"),
            **settings,
        )

    # ---- lifecycle ----

    def _reset(self) -> None:
        self._hold_arm: Optional[np.ndarray] = None
        self._ramp: tuple[np.ndarray, ...] = ()
        self._ramp_index = 0
        self._start_measured: Optional[np.ndarray] = None
        # 始めから物を握っていた側 (左, 右)。_begin_ramp で決める。
        self._holding = np.zeros(2, dtype=bool)
        # 動かして到達を見る側。release では離す手だけ (他の手は見ない)。
        self._watched = np.ones(2, dtype=bool)
        self._target = self._configured_target.copy()
        self._history: list[np.ndarray] = []
        self._stable_samples = 0
        # 締め切り・watchdog・ramp は「指令を進められた制御時間」で数える
        # (Issue #159 T6)。1 tick 最大 MAX_TICK_CONTROL_SEC。
        self._control_elapsed = 0.0
        self._last_at: Optional[float] = None
        self._ramp_sent_index = -1
        self._deadline: Optional[float] = None
        self._complete = False
        self._completion_mode: Optional[str] = None
        self._failure_reason: Optional[str] = None
        # 実測は来ているが締め切りまでに目標へ届かなかった (故障ではない)。
        self._timeout_reason: Optional[str] = None
        self._ramp_started_at: Optional[float] = None

    def _on_start(self, params: dict) -> None:
        self._reset()
        # 測定値が来るまで ramp を始められないので、まず「待つ期限」を張る。
        # 測定が来たら _begin_ramp が移動時間に応じた期限に張り替える。これが無いと
        # Dex1 の state が途切れたときに完了も失敗もせず、腕を持ったまま止まらない。
        self._deadline = self._minimum_timeout_sec

    def _on_stop(self) -> None:
        pass

    # ---- per-tick ----

    def _advance_control_time(self) -> None:
        now = self._time_fn()
        if self._last_at is not None:
            self._control_elapsed += min(
                max(now - self._last_at, 0.0), MAX_TICK_CONTROL_SEC
            )
        self._last_at = now

    def step(self, obs: dict) -> np.ndarray:
        health_check = getattr(self._hand_actuator, "raise_if_unhealthy", None)
        if health_check is not None:
            health_check()
        self._advance_control_time()
        if self._hold_arm is None:
            target = (
                self._hold_arm_target_provider()
                if self._hold_arm_target_provider is not None else None
            )
            self._hold_arm = (
                _measured_arm_14(obs) if target is None
                else np.asarray(target, dtype=np.float64).copy()
            )
            if self._hold_arm.shape != (14,) or not np.isfinite(self._hold_arm).all():
                raise RuntimeError(f"{self.name} cannot hold an invalid arm target")
        measured = self._measured_hand(obs)
        if measured is not None and self._start_measured is None:
            self._begin_ramp(measured)
        if self._ramp:
            # ramp は control_hz の段で作ってある。1 tick に最低 1 段 (30 Hz 以上は
            # 今までどおり)、tick が遅いときは制御時間の分まで進めるので、速さ
            # (velocity_limit_rad_s) は変わらない (Issue #159 T6)。
            assert self._ramp_started_at is not None
            due = int(
                math.floor(
                    (self._control_elapsed - self._ramp_started_at) * self._control_hz
                    + 1e-6
                )
            )
            self._ramp_sent_index = min(max(due, self._ramp_sent_index + 1), len(self._ramp) - 1)
            command = self._ramp[self._ramp_sent_index]
            self._ramp_index = self._ramp_sent_index + 1
            self._hand_actuator.send_action(command)
        if measured is not None and not self._complete:
            self._update_completion(measured)
            self._check_motion_watchdog(measured)
        if not self._complete:
            self._check_deadline()
        return self._hold_arm.copy()

    def _measured_hand(self, obs: dict) -> Optional[np.ndarray]:
        """新しくて有限な Dex1 の開度。古い / 未受信なら None。"""
        state = obs.get("hand_state")
        if state is None:
            return None
        now_ns = time.monotonic_ns()
        age_s = max(
            now_ns - int(state.left_received_monotonic_ns),
            now_ns - int(state.right_received_monotonic_ns),
        ) / 1e9
        measured = np.asarray(state.position_rad, dtype=np.float64)
        if age_s > self._state_max_age_s or measured.shape != (2,) or not np.isfinite(measured).all():
            self._history.clear()
            self._stable_samples = 0
            return None
        if bool(getattr(state, "health_diagnostics_available", False)):
            faults = np.asarray(getattr(state, "fault_code", (0, 0)), dtype=np.uint32)
            if faults.shape != (2,):
                self._failure_reason = (
                    f"{self.name} received malformed Dex1 fault diagnostics: "
                    f"shape={faults.shape}"
                )
                return None
            failed = np.flatnonzero(faults)
            if failed.size:
                sides = ("left", "right")
                details = []
                for index in failed:
                    code = int(faults[index])
                    names = [
                        text
                        for bit, text in DEX1_FAULT_DESCRIPTIONS.items()
                        if code & bit
                    ]
                    details.append(
                        f"{sides[index]}=0x{code:x}"
                        + (f" ({', '.join(names)})" if names else "")
                    )
                self._failure_reason = (
                    f"{self.name} refused to command a faulted Dex1 motor: "
                    f"{'; '.join(details)}. Clear the physical motor fault (a latched "
                    "thermal fault requires a complete Dex1 motor-power cycle), then "
                    "verify with smoke_hand_real before retrying"
                )
                return None
        return measured

    def _last_commanded(self) -> Optional[np.ndarray]:
        """actuator が最後に出した指令。読めない / 不正なら None。

        不正な値は `_update_completion` が failure にするので、ここでは使わないだけ。
        """
        read_commanded = getattr(
            self._hand_actuator, "read_last_commanded_positions", None
        )
        if not callable(read_commanded):
            return None
        commanded = read_commanded()
        if commanded is None:
            return None
        array = np.asarray(commanded, dtype=np.float64)
        if array.shape != (2,) or not np.isfinite(array).all():
            return None
        return array

    def _begin_ramp(self, measured: np.ndarray) -> None:
        self._start_measured = measured.copy()
        commanded = self._last_commanded()
        # 指令を途切れさせない: 直前の指令から目標へ (Issue #159 B4a-04)
        ramp_start = measured if commanded is None else commanded
        if self._release and not self._select_release(measured, commanded, ramp_start):
            return
        if commanded is not None and self._allow_closing_contact:
            self._holding = (
                (measured - commanded > self._holding_contact_gap_rad)
                & (self._target < measured - self._tolerance_rad)
            )
        self._ramp = build_hand_target_ramp(
            ramp_start,
            self._target,
            command_hz=self._control_hz,
            velocity_limit_rad_s=self._velocity_limit_rad_s,
        )
        self._ramp_index = 0
        self._ramp_sent_index = -1
        self._ramp_started_at = self._control_elapsed
        travel_s = float(
            np.max(np.abs(self._target - np.vstack((measured, ramp_start))))
        ) / self._velocity_limit_rad_s
        self._deadline = self._control_elapsed + max(
            self._minimum_timeout_sec, travel_s + self._timeout_margin_sec
        )
        holding = (
            f", holding={self._holding.tolist()}" if np.any(self._holding) else ""
        )
        print(
            f"[hand] {self.name}: {measured.tolist()} -> {self._target.tolist()} "
            f"(from command {ramp_start.tolist()}, {len(self._ramp)} steps{holding})",
            file=sys.stderr,
        )

    def _select_release(
        self,
        measured: np.ndarray,
        commanded: Optional[np.ndarray],
        ramp_start: np.ndarray,
    ) -> bool:
        """離す手を決める。離す手が無ければ完了にして False (Issue #159 T3)。"""
        can_hold = measured >= self._holding_min_opening_rad
        if commanded is None:
            gripping = can_hold
        else:
            gripping = can_hold & (
                measured - commanded > self._holding_contact_gap_rad
            )
        # 離さない手は直前の指令のまま (目標 = 今の指令なので ramp も動かない)
        self._target = np.where(gripping, self._configured_target, ramp_start)
        self._watched = gripping
        if np.any(gripping):
            return True
        self._complete = True
        self._completion_mode = "nothing_to_release"
        print(
            f"[hand] {self.name}: nothing to release (measured={measured.tolist()}, "
            f"command={None if commanded is None else commanded.tolist()})",
            file=sys.stderr,
        )
        return False

    def _check_motion_watchdog(self, measured: np.ndarray) -> None:
        """Fail early when fresh state is being republished but hardware is frozen."""

        if (
            self._failure_reason is not None
            or self._complete
            or self._start_measured is None
            or self._ramp_started_at is None
            or self._control_elapsed - self._ramp_started_at < NO_MOTION_WATCHDOG_SEC
            or np.max(np.abs(self._target - self._start_measured)) <= self._tolerance_rad
        ):
            return
        requested_motion = np.abs(self._target - self._start_measured)
        progress = np.abs(measured - self._start_measured)
        # 握っていた側は動かないのが正常なので見ない (Issue #159 B4a-04)
        stalled = np.flatnonzero(
            (requested_motion > self._tolerance_rad)
            & (progress < NO_MOTION_MIN_PROGRESS_RAD)
            & ~self._holding
            & self._watched
        )
        if stalled.size:
            side_names = ("left", "right")
            stalled_detail = ", ".join(
                f"{side_names[index]}(requested={requested_motion[index]:.4f}rad, "
                f"progress={progress[index]:.4f}rad)"
                for index in stalled
            )
            self._failure_reason = (
                f"{self.name} received fresh Dex1 state but the following hardware "
                f"did not move after {NO_MOTION_WATCHDOG_SEC:.1f}s: {stalled_detail}. "
                "Check the Dex1 command path: on the self path the Orin "
                "dex1_1_gripper_server serial link and both Dex1 USB motors; on the "
                "competition path the organizer's run_wbc_with_dex1 and the adapter's "
                "--dex1-port"
            )

    def _update_completion(self, measured: np.ndarray) -> None:
        self._history.append(measured.copy())
        self._history = self._history[-STABLE_HISTORY_LEN:]
        # Reaching the measured pose is insufficient when either the 30 Hz
        # staging ramp or the actuator's 200 Hz publisher-side slew limiter is
        # still moving.  In particular, contact used to complete early while
        # the final frame-zero command had not yet been emitted.
        command_complete = self._ramp_index >= len(self._ramp)
        read_commanded = getattr(
            self._hand_actuator, "read_last_commanded_positions", None
        )
        if callable(read_commanded):
            commanded = read_commanded()
            if commanded is None:
                command_complete = False
            else:
                commanded_array = np.asarray(commanded, dtype=np.float64)
                if (
                    commanded_array.shape != (2,)
                    or not np.isfinite(commanded_array).all()
                ):
                    self._failure_reason = (
                        f"{self.name} received invalid Dex1 command telemetry: "
                        f"{commanded!r}"
                    )
                    return
                command_complete = command_complete and bool(
                    np.max(np.abs(commanded_array - self._target))
                    <= self._tolerance_rad
                )
        mode = hand_target_completion_mode(
            start=self._start_measured,
            target=self._target,
            measured_history=self._history,
            tolerance_rad=self._tolerance_rad,
            allow_closing_contact=self._allow_closing_contact,
            holding=self._holding,
            active=self._watched,
        )
        self._stable_samples = (
            self._stable_samples + 1
            if mode == "target" and command_complete
            else 0
        )
        if command_complete and (
            self._stable_samples >= self._required_stable_samples
            or mode in ("closing_contact", "holding_contact")
        ):
            self._complete = True
            self._completion_mode = mode
            print(
                f"[hand] {self.name}: {mode} (measured={measured.tolist()}, "
                f"residual={np.abs(measured - self._target).tolist()})",
                file=sys.stderr,
            )

    def _check_deadline(self) -> None:
        if self._deadline is None or self._control_elapsed < self._deadline:
            return
        if self._start_measured is None:
            # 締め切りまで Dex1 の実測が一度も来なかった = センサーが来ていない故障。
            self._failure_reason = (
                f"{self.name} received no fresh Dex1 state before the deadline"
            )
            return
        # 実測は来ていて、目標にだけ届かなかった。時間切れとして次へ進ませる。
        measured = "unknown" if not self._history else self._history[-1].tolist()
        self._timeout_reason = (
            f"{self.name} did not reach {self._target.tolist()} before the deadline "
            f"(measured={measured})"
        )

    # ---- orchestrator が読む状態 ----

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def failure_reason(self) -> Optional[str]:
        """故障の理由 (Dex1 の故障・全く動かない・実測が来ない等)。orchestrator が止める。"""
        return self._failure_reason

    @property
    def timeout_reason(self) -> Optional[str]:
        """締め切りまでに目標へ届かなかった理由。orchestrator は時間切れとして次へ進む。"""
        return self._timeout_reason

    @property
    def completion_mode(self) -> Optional[str]:
        """完了の仕方 (`target` / `closing_contact` / `holding_contact` /
        `nothing_to_release`)。記録用。"""
        return self._completion_mode
