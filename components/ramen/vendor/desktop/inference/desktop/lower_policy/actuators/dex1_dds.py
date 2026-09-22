"""Dex1-1 グリッパの DDS 指令 / 状態経路 (Issue #123)。

## 何が無かったか

本 Issue 以前、repo には **グリッパへ指令を送る経路が存在しなかった**。
`inference/desktop/xr/check_dex1_state.py` は冒頭で
"creates subscribers only: it never creates a publisher and therefore cannot
command either gripper" と明記しており、`actuators/` にも hand actuator は無い。
状態 topic (`rt/dex1/{side}/state`) は存在するが `obs` にも乗っていない。

## 構成

`Dex1Gripper` Protocol の実装。**skill が自前で保持する** ことで
`Orchestrator._build_obs` を変更せずに済ませる (`MoveToTable` が walk actuator を
直接叩く Type A の流儀と同じ)。

- 指令: `rt/dex1/{side}/cmd` へ `MotorCmds_` を publish
- 状態: `rt/dex1/{side}/state` から `MotorStates_` を latest-only で保持

cyclonedds / unitree_sdk2py は `__init__` 内で lazy import する (CLAUDE.md の
runtime env 方針)。default env から本 module を import しても、instantiate しなければ
ImportError にならない。

## 前提

Orin / PC2 側で `dex1_1_gripper_server` が起動していること。起動していないと
状態 topic が来ず `read()` が `None` を返し続ける (skill 側は把持判定不能として
timeout する)。

## test について

publisher / subscriber を constructor で inject できるようにしてあり、
SDK 無しで payload 構築 / 安全クランプ / lifecycle を検証できる
(`actuators/g1_arm_sdk.py` と同じ流儀)。
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from inference.desktop.lower_policy.kinematics.types import Side

# Orin 側 dex1_1_gripper_server の topic 名。
DEFAULT_CMD_TOPIC: dict[Side, str] = {
    Side.LEFT: "rt/dex1/left/cmd",
    Side.RIGHT: "rt/dex1/right/cmd",
}
DEFAULT_STATE_TOPIC: dict[Side, str] = {
    Side.LEFT: "rt/dex1/left/state",
    Side.RIGHT: "rt/dex1/right/state",
}


#: 学習データの最速グリッパ速度 (unit/s、0=全閉 / 4.5=全開 の空間)。
#:
#: 出所: `BitRobot/G1_WBT_Dex1_Building-Children-Table` の
#: `observation.state.hand_state` を episode 0 の 153 秒 / 4600 frame で実測し、
#: 0.30 秒窓の平均速度の最大を取った値 (2026-09-22、Issue #154)。
#:
#: ```
#: 左手  max 4.20  p99 3.46  p95 3.08 unit/s
#: 右手  max 3.34  p99 3.29  p95 3.00 unit/s
#: ```
#:
#: **なぜ制限するか。** 自前機は制限が無く、実測 5.43 unit/s (= 6.4 motor rad/s、
#: WandB `pick_table_leg_*_20260920` の `hand_position_rad`) で動いており、
#: **学習データより 1.3 倍速い**。policy は学習時の速度で腕とグリッパの
#: タイミングを学んでいるので、速すぎる側もずれている。
#:
#: 会場側は運営の `run_wbc_with_dex1.py --dex1-max-speed` が既定 2.0 motor rad/s
#: (= 1.70 unit/s) で slew しており、**学習の 2.5 分の 1**。そちらは我々が渡す
#: 引数なので 09-27 で擦り合わせる (`docs/handoff/venue_runbook.md`)。
#:
#: ⚠️ 実測は episode 0 の 153 秒のみ。窓幅 0.30 秒は運営の slew が効く時間尺度に
#: 合わせたもので、窓を変えれば値も動く。広げる場合はこの定数も見直すこと。
TRAIN_MAX_SPEED_UNITS_PER_S: float = 4.2


@dataclass(frozen=True)
class Dex1Limits:
    """指令値の安全枠。0 = 全閉 / 4.5 = 全開。

    設定ミスで機械限界を超える指令が飛ぶのを防ぐため、`command` で必ずクランプする。

    Attributes:
        max_speed: 位置指令の最大変化率 (unit/s)。`None` で無制限。
            既定は学習データの実測最速 (`TRAIN_MAX_SPEED_UNITS_PER_S`)。
            運営の `Dex1Injector.apply` と同じく **最初の 1 指令だけは即座に採る**
            (目標が無い状態からの slew には意味が無いため)。
    """

    min_position: float = 0.0
    max_position: float = 4.5
    max_speed: Optional[float] = TRAIN_MAX_SPEED_UNITS_PER_S

    def __post_init__(self) -> None:
        if not self.min_position < self.max_position:
            raise ValueError("Dex1Limits: min_position < max_position を満たすこと")
        if self.max_speed is not None and (
            not math.isfinite(self.max_speed) or self.max_speed <= 0.0
        ):
            raise ValueError("Dex1Limits: max_speed must be positive finite or None")

    def clamp(self, position: float) -> float:
        value = float(position)
        if not math.isfinite(value):
            raise ValueError("Dex1 position must be finite")
        return float(min(max(value, self.min_position), self.max_position))


@dataclass(frozen=True)
class Dex1Gains:
    """位置制御のゲイン。把持は「全閉指令で脚に機械的に止めさせる」ので、
    握り込みすぎない程度の弱めの kp にしておく。

    **値は運営の reference calibration に合わせてある** (2026-09-22、Issue #154):

        tools/diagnose_dex1.py
          "(Reference calibration for one unit, if H1: indices 31/33,
            q=0.0 closed, q=-5.30 open, kp=5.0, kd=0.05 -- sign-flipped
            from Unitree's reference, so **trust these, not the docs**.)"
        tools/run_wbc_with_dex1.py
          GRIPPER_KP = 5.0 / GRIPPER_KD = 0.05   ← 会場で実際に書き込まれる値

    会場のグリッパは運営 injector が `rt/lowcmd` の slot 31/33 に自分の kp/kd を
    書くので、ここは **自前機だけ**に効く。合わせる理由は「自前機の評価が会場の
    挙動を予測できるようにする」こと。

    ⚠️ kd はかつて 0.1 だった (2026-09-11 の暫定値、docstring も「実機で要調整」と
    書いたまま一度も調整されていなかった)。速度 (`TRAIN_MAX_SPEED_UNITS_PER_S`) と
    違って **ゲインは学習が持っている量ではない** — 学習データ側の値は分からず
    設定もできないので、会場の値が唯一の実行可能なアンカー。
    """

    kp: float = 5.0
    kd: float = 0.05


class Dex1DdsGripper:
    """`Dex1Gripper` Protocol の DDS 実装。

    Args:
        limits / gains: 安全枠と制御ゲイン。
        cmd_topic / state_topic: topic 名の上書き。
        publisher_factory: `(topic) -> publisher` を返す注入フック。test 用。
            None のとき SDK を lazy import して実 publisher を作る。
        subscriber_factory: `(topic, handler) -> subscriber` を返す注入フック。
        cmd_factory: `() -> MotorCmds_` を返す注入フック。
    """

    def __init__(
        self,
        *,
        limits: Optional[Dex1Limits] = None,
        gains: Optional[Dex1Gains] = None,
        cmd_topic: Optional[dict[Side, str]] = None,
        state_topic: Optional[dict[Side, str]] = None,
        publisher_factory: Optional[Callable[[str], Any]] = None,
        subscriber_factory: Optional[
            Callable[[str, Callable[[Any], None]], Any]
        ] = None,
        cmd_factory: Optional[Callable[[], Any]] = None,
        continuous_publish_hz: Optional[float] = 200.0,
    ) -> None:
        self._limits = limits or Dex1Limits()
        self._gains = gains or Dex1Gains()
        self._cmd_topic = dict(cmd_topic or DEFAULT_CMD_TOPIC)
        self._state_topic = dict(state_topic or DEFAULT_STATE_TOPIC)
        for side in (Side.LEFT, Side.RIGHT):
            if side not in self._cmd_topic or side not in self._state_topic:
                raise ValueError(
                    f"topic mapping must cover both sides (missing {side})"
                )

        self._lock = threading.Lock()
        self._state: dict[Side, Optional[float]] = {Side.LEFT: None, Side.RIGHT: None}
        #: 呼び出し側が指令した **目標値**。`last_command` はこれを返す。
        self._last_cmd: dict[Side, Optional[float]] = {
            Side.LEFT: None,
            Side.RIGHT: None,
        }
        #: 実際に publish する **追従中の値**。`limits.max_speed` で目標へ寄せる。
        self._current: dict[Side, Optional[float]] = {
            Side.LEFT: None,
            Side.RIGHT: None,
        }
        #: 側ごとの直近 slew 時刻 (monotonic)。dt はここから測る。
        self._slew_t: dict[Side, Optional[float]] = {
            Side.LEFT: None,
            Side.RIGHT: None,
        }
        self._closed = False
        if continuous_publish_hz is not None and (
            not math.isfinite(continuous_publish_hz) or continuous_publish_hz <= 0.0
        ):
            raise ValueError("continuous_publish_hz must be positive finite or None")
        self._continuous_publish_hz = continuous_publish_hz
        self._stop_event = threading.Event()
        self._publisher_thread: Optional[threading.Thread] = None
        self._publisher_error: Optional[BaseException] = None

        if (
            publisher_factory is None
            or subscriber_factory is None
            or cmd_factory is None
        ):
            # SDK は instantiate 時に初めて import する (default env での import 保護)。
            try:
                from unitree_sdk2py.core.channel import (  # type: ignore
                    ChannelPublisher,
                    ChannelSubscriber,
                )
                from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # type: ignore
                    MotorCmd_,
                    MotorCmds_,
                    MotorStates_,
                )
            except ImportError as exc:  # pragma: no cover - runtime env でのみ発生
                raise ImportError(
                    "Dex1DdsGripper needs unitree_sdk2py (runtime feature-env). "
                    "`pixi run -e runtime ...` で実行するか、test では "
                    "publisher_factory / subscriber_factory / cmd_factory を inject すること"
                ) from exc

            if publisher_factory is None:

                def publisher_factory(topic: str) -> Any:  # noqa: F811
                    pub = ChannelPublisher(topic, MotorCmds_)
                    pub.Init()
                    return pub

            if subscriber_factory is None:

                def subscriber_factory(
                    topic: str, handler: Callable[[Any], None]
                ) -> Any:  # noqa: F811
                    sub = ChannelSubscriber(topic, MotorStates_)
                    sub.Init(handler, 10)
                    return sub

            if cmd_factory is None:

                def cmd_factory() -> Any:  # noqa: F811
                    # The SDK default MotorCmds_ helper has an empty sequence;
                    # Dex1 is one motor per side and therefore needs one
                    # explicitly initialized MotorCmd_.
                    return MotorCmds_(
                        cmds=[
                            MotorCmd_(
                                mode=1,
                                q=0.0,
                                dq=0.0,
                                tau=0.0,
                                kp=self._gains.kp,
                                kd=self._gains.kd,
                                reserve=[0, 0, 0],
                            )
                        ]
                    )

        self._cmd_factory = cmd_factory
        self._publishers: dict[Side, Any] = {
            side: publisher_factory(self._cmd_topic[side])
            for side in (Side.LEFT, Side.RIGHT)
        }
        self._subscribers: dict[Side, Any] = {
            side: subscriber_factory(self._state_topic[side], self._make_handler(side))
            for side in (Side.LEFT, Side.RIGHT)
        }

    # ------------------------------------------------------------ 状態

    def _make_handler(self, side: Side) -> Callable[[Any], None]:
        def _handler(msg: Any) -> None:
            position = self._extract_position(msg)
            if position is None:
                return
            with self._lock:
                self._state[side] = position

        return _handler

    @staticmethod
    def _extract_position(msg: Any) -> Optional[float]:
        """`MotorStates_` から 1 自由度分の位置を取り出す。

        壊れた / 空の sample は握り潰して `None` を返す (制御ループを例外で
        落とさない。未受信と同じ扱いになり、skill 側は timeout で検知する)。
        """
        states = getattr(msg, "states", None)
        if not states:
            return None
        q = getattr(states[0], "q", None)
        if q is None:
            return None
        try:
            value = float(q)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def read(self, side: Side) -> Optional[float]:
        self._raise_publisher_error()
        with self._lock:
            return self._state[side]

    def last_command(self, side: Side) -> Optional[float]:
        """呼び出し側が指令した **目標値**。未送信なら `None`。

        ⚠️ `max_speed` が有効なとき、実際に publish している値はこれとは違う
        (目標へ向かって追従中)。**把持判定 (`classify_grasp`) にこれを使うと、
        閉じ始めの過渡で「指令より開いた位置で止まっている」ように見えて
        誤って HOLDING になる。** その用途には `current_command` を使うこと。
        """
        with self._lock:
            return self._last_cmd[side]

    def current_command(self, side: Side) -> Optional[float]:
        """いま実際に publish している値 (slew 後)。未送信なら `None`。

        `classify_grasp` の `command` 引数にはこちらを渡す。実測 (`read`) が
        追従しようとしている相手はこの値であって、最終目標ではない。
        """
        with self._lock:
            return self._current[side]

    # ------------------------------------------------------------ 指令

    def build_command(self, position: float) -> Any:
        """publish する `MotorCmds_` を組む (test から直接呼べるよう分離)。"""
        msg = self._cmd_factory()
        cmds = getattr(msg, "cmds", None)
        if not cmds:
            raise RuntimeError("cmd_factory returned a message without 'cmds'")
        motor = cmds[0]
        motor.q = float(position)
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = self._gains.kp
        motor.kd = self._gains.kd
        return msg

    def _slew(self, side: Side, target: float) -> float:
        """`target` へ `max_speed` で寄せた、いま publish すべき値を返す。

        運営の `tools/run_wbc_with_dex1.py:118-140` (`Dex1Injector.apply`) と同じ形:
        経過時間から `step = max_speed * dt` を作り、目標との差をその幅で clip する。
        **最初の 1 指令だけは目標をそのまま採る** — 追従元が無い状態で slew しても
        意味が無いうえ、起動時に必ず全開から始まる想定を壊すため。

        呼び出し側は `self._lock` を保持していること。
        """
        max_speed = self._limits.max_speed
        now = time.monotonic()
        current = self._current[side]
        if max_speed is None or current is None:
            self._current[side] = target
            self._slew_t[side] = now
            return target
        last = self._slew_t[side]
        # dt の上限は 0.1s。スレッド停止後などに 1 回で飛ばさないため (運営も同じ)。
        dt = 0.02 if last is None else min(max(now - last, 1e-3), 0.1)
        step = max_speed * dt
        moved = current + min(max(target - current, -step), step)
        self._current[side] = moved
        self._slew_t[side] = now
        return moved

    def command(self, side: Side, position: float) -> None:
        if self._closed:
            raise RuntimeError("Dex1DdsGripper is closed")
        self._raise_publisher_error()
        clamped = self._limits.clamp(position)
        with self._lock:
            self._last_cmd[side] = clamped
            to_send = self._slew(side, clamped)
        self._publishers[side].Write(self.build_command(to_send))
        self._ensure_publisher_thread()

    def _ensure_publisher_thread(self) -> None:
        """Start the official-rate hold publisher after the first real command.

        Constructing this object is intentionally non-actuating: no background
        writer starts until ``command`` has completed one successful DDS write.
        Thereafter the latest per-side target is re-published at 200 Hz, matching
        the repository's canonical ``G1HandActuator`` and the Unitree Dex1
        controller.  This avoids a transient DDS packet leaving a gripper with
        no maintained target during the multi-second arm stages.
        """
        if self._continuous_publish_hz is None or self._publisher_thread is not None:
            return
        thread = threading.Thread(
            target=self._publisher_loop,
            name="dex1-rule-based-publisher",
            daemon=False,
        )
        self._publisher_thread = thread
        thread.start()

    def _publisher_loop(self) -> None:
        assert self._continuous_publish_hz is not None
        period = 1.0 / self._continuous_publish_hz
        deadline = time.monotonic() + period
        while not self._stop_event.wait(max(0.0, deadline - time.monotonic())):
            # 保持 publish も slew を通す。ここを通さないと 200Hz の hold が
            # 目標値をそのまま出し続け、`command` 側の rate limit が無意味になる。
            with self._lock:
                to_send = {
                    side: (None if target is None else self._slew(side, target))
                    for side, target in self._last_cmd.items()
                }
            try:
                for side, value in to_send.items():
                    if value is not None:
                        self._publishers[side].Write(self.build_command(value))
            except BaseException as exc:  # surfaced synchronously on the next tick
                with self._lock:
                    self._publisher_error = exc
                self._stop_event.set()
                return
            deadline += period
            now = time.monotonic()
            if deadline < now - period:
                # Never burst stale writes after scheduler starvation.
                deadline = now + period

    def _raise_publisher_error(self) -> None:
        with self._lock:
            error = self._publisher_error
        if error is not None:
            raise RuntimeError("Dex1 continuous publisher failed") from error

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """subscriber / publisher を teardown する (idempotent)。

        `Close()` を持たない実装もあるので存在するときだけ呼ぶ。
        native thread を残すと teardown 時に segfault しうる (Issue #101 と同種)。
        """
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        thread = self._publisher_thread
        thread_failed_to_stop = False
        if thread is not None:
            thread.join(timeout=1.0)
            thread_failed_to_stop = thread.is_alive()
        for holder in (self._subscribers, self._publishers):
            for handle in holder.values():
                closer = getattr(handle, "Close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:  # pragma: no cover - teardown は落とさない
                        pass
        if thread_failed_to_stop:
            raise RuntimeError("Dex1 continuous publisher thread did not stop")

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------ config

    @classmethod
    def from_config(cls, cfg: Optional[dict], **kwargs: Any) -> "Dex1DdsGripper":
        """YAML section から構築する。

        期待する構造 (全て任意):
            {"limits": {"min_position": .., "max_position": ..},
             "gains": {"kp": .., "kd": ..}}
        """
        cfg = cfg or {}
        if not isinstance(cfg, dict):
            raise ValueError("dex1: must be a mapping")
        unknown = set(cfg) - {"limits", "gains"}
        if unknown:
            raise ValueError(f"dex1: unknown key(s) {sorted(unknown)}")
        # `or {}` で正規化してはいけない: 空 list / 空文字が falsy として
        # 素通りし、型の誤りを見逃す。None だけを default 扱いする。
        sections: dict[str, dict] = {}
        for name, allowed in (
            ("limits", {"min_position", "max_position"}),
            ("gains", {"kp", "kd"}),
        ):
            sub = cfg.get(name)
            if sub is None:
                sections[name] = {}
                continue
            if not isinstance(sub, dict):
                raise ValueError(f"dex1.{name}: must be a mapping")
            extra = set(sub) - allowed
            if extra:
                raise ValueError(f"dex1.{name}: unknown key(s) {sorted(extra)}")
            sections[name] = sub
        limits_cfg, gains_cfg = sections["limits"], sections["gains"]
        return cls(
            limits=Dex1Limits(**{k: float(v) for k, v in limits_cfg.items()}),
            gains=Dex1Gains(**{k: float(v) for k, v in gains_cfg.items()}),
            **kwargs,
        )
