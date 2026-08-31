"""Orchestrator: YOLO-OBB streaming state-machine の tick pipeline 本体。

Issue #47 の core module。既存 parts を束ねて 1 frame = 1 tick で回す:

    FrameSource → YoloObbPerception → DetectionStream → SkillState.update
        → TRANSITIONS graph で enter_check → 発火なら SkillState.transition + Dispatcher.start
        → Dispatcher.step で per-tick action → (Type B なら) actuator に送信 → JSONL log

完全な forward streaming で、real-time (ROS2 source) と replay (Lerobot source)
を同じ code path で回す。
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, TextIO

import numpy as np

from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
from inference.desktop.perception.frame_source import FrameData, FrameSource
from inference.desktop.perception.dex1_state_source import Dex1StateData
from inference.desktop.perception.joint_state_source import JointStateData
from inference.desktop.perception.stream import DetectionStream
from inference.desktop.perception.yolo_obb import OBBDetection
from inference.desktop.skill_planner.enter_conditions import (
    enter_flip_table,
    enter_insert_table_leg,
    enter_move_table_base,
    enter_pick_table_leg,
    enter_rotate_leg_to_tighten,
)
from inference.desktop.skill_planner.geometry import mean_verts_pivot_aligned
from inference.desktop.skill_planner.state import SkillState


# =========================================================================
# module 定数: default transition graph + enter_check registry
# =========================================================================
DEFAULT_TRANSITIONS: dict[str, list[str]] = {
    "setup":                 ["move_to_table"],  # Issue #81 Phase 3: 腕 pre-motion
    "move_to_table":         ["move_table_base"],
    "move_table_base":       ["pick_table_leg"],
    "pick_table_leg":        ["insert_table_leg"],
    "insert_table_leg":      ["rotate_leg_to_tighten"],
    "rotate_leg_to_tighten": ["flip_table", "move_table_base"],  # ★ priority = list 順
    "flip_table":            [],
}

def _enter_never(dets: list[OBBDetection], state: SkillState) -> bool:
    """timer (max_dwell_sec) のみで進む skill 用の enter_check。YOLO 検出では fire しない。

    setup / move_to_table は前段の完了 (=時間) で次段へ進むため、enter_check では常に
    False を返す。tick() は enter_check[cand] を直接 index するので、dwell-only の遷移先も
    registry に entry が要る (未登録だと KeyError で即死する。Issue #97)。
    """
    return False


DEFAULT_ENTER_CHECK: dict[str, Callable[[list[OBBDetection], SkillState], bool]] = {
    "move_to_table":         _enter_never,   # dwell (max_dwell_sec) のみで move_table_base へ
    "move_table_base":       enter_move_table_base,
    "pick_table_leg":        enter_pick_table_leg,
    "insert_table_leg":      enter_insert_table_leg,
    "rotate_leg_to_tighten": enter_rotate_leg_to_tighten,
    "flip_table":            enter_flip_table,
}


# =========================================================================
# Phase 3 stage-based orchestration (Issue #128)
# =========================================================================
#
# 各 --stage=N は「その stage 内で dispatch される skill 列」を宣言する。
# stage 0 = 準備 (歩く + 腕 pre-motion + rotate_table_base 初期姿勢)
# stage 1-4 = 各 leg round (rotate_table_base → pick → insert → rotate_leg_to_tighten)
# stage 5 = 最終 (flip_table)
#
# 各 stage は独立に起動可能。前 stage 終了状態が current state と一致してれば
# --stage=N で再開できる。stage 内の transition graph は
# build_stage_transitions() で自動生成する。
#
# skill_name の慣例:
#   arm_pre_motion_for_rotate_table_base = CollisionAwareArmPreMotionSkill
#       (final_pose = skill_config.yaml:skills.rotate_table_base.initial_pose)
#   arm_pre_motion_for_flip_table = CollisionAwareArmPreMotionSkill
#       (final_pose = skill_config.yaml:skills.flip_table.initial_pose)

STAGE_SKILL_SEQUENCES: dict[int, list[str]] = {
    # Stage 0 ends with the arms still in the measured lowered pose.  The
    # first learned expert is loaded while that safe pose is held, then Stage
    # 1 performs the collision-aware pre-motion in the same control process.
    0: ["setup", "move_to_table", "post_walk_settle"],
    1: ["arm_pre_motion_for_rotate_table_base", "rotate_table_base",
        "pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"],
    2: ["arm_pre_motion_for_rotate_table_base", "rotate_table_base",
        "pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"],
    3: ["arm_pre_motion_for_rotate_table_base", "rotate_table_base",
        "pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"],
    4: ["arm_pre_motion_for_rotate_table_base", "rotate_table_base",
        "pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"],
    5: ["arm_pre_motion_for_flip_table", "flip_table"],
}
STAGE_MIN: int = 0
STAGE_MAX: int = 5


def build_stage_transitions(stage: int) -> dict[str, list[str]]:
    """Stage の skill 列を SkillDispatchLowerPolicy 用の transition graph に展開。

    stage の最後の skill は terminal (empty list) = orchestrator が
    is_complete で run_live を抜ける経路に載せる。stage 内の各 skill は
    次 skill 1 個だけを候補に持つ (list length 1)。

    Args:
        stage: STAGE_SKILL_SEQUENCES key (0..5)。

    Returns:
        {skill_name: [next_skill_name] or []} の dict。

    Raises:
        KeyError: stage が STAGE_SKILL_SEQUENCES に無い。
    """
    if stage not in STAGE_SKILL_SEQUENCES:
        raise KeyError(
            f"stage {stage} not registered (valid: {sorted(STAGE_SKILL_SEQUENCES)})"
        )
    skills = STAGE_SKILL_SEQUENCES[stage]
    transitions: dict[str, list[str]] = {}
    for i, skill in enumerate(skills):
        transitions[skill] = [skills[i + 1]] if i + 1 < len(skills) else []
    return transitions


# =========================================================================
# Perception protocol (duck typing、YoloObbPerception も stub も受ける)
# =========================================================================
class PerceptionProtocol(Protocol):
    """Orchestrator が要求する perception layer の最小 interface。

    YoloObbPerception は自然に satisfy する。test 側で stub を渡す用途にも使う。
    """

    def predict(self, rgb: np.ndarray) -> list[OBBDetection]: ...


class JointStateSourceProtocol(Protocol):
    """Orchestrator が要求する joint state source の最小 interface。

    `JointStateSource` は自然に satisfy する。obs に joint state を乗せない
    運用 (offline replay 等) では None を渡して disable する。
    """

    def get(self) -> Optional[JointStateData]: ...


class Dex1StateSourceProtocol(Protocol):
    """Read-only bilateral Dex1 state source used by learned policies."""

    def get(self) -> Optional[Dex1StateData]: ...


# =========================================================================
# TickResult
# =========================================================================
@dataclass(frozen=True)
class TickResult:
    """1 tick の出力 (debug / test / log 用)。"""

    t: int
    current_skill: str
    fire_transition_to: Optional[str]  # transition 起きたら新 skill 名、無ければ None
    cleaned: list[OBBDetection]
    action: Optional[np.ndarray]        # dispatcher.step の返り値 (Type B の action tensor)


class LiveSourceSafetyError(RuntimeError):
    """Live source の startup / freshness / initial skill timeout。"""


# =========================================================================
# helpers
# =========================================================================
def _table_top_verts(dets: list[OBBDetection]) -> Optional[np.ndarray]:
    """confidence 最大の table_top OBB の verts。無検出は None。

    enter_conditions.py にも同名 private helper があるが、reuse は現状 2 箇所のため
    共通化せず独立実装 (YAGNI: 3rd reuse で geometry 側に切り出し検討)。
    """
    tops = [d for d in dets if d.class_name == "table_top"]
    if not tops:
        return None
    top = max(tops, key=lambda d: d.confidence)
    return top.verts


# =========================================================================
# Orchestrator
# =========================================================================
class Orchestrator:
    """YOLO-OBB streaming state-machine の tick pipeline 本体。

    Args:
        perception: predict(rgb) → list[OBBDetection] の Protocol 実装。
        cleaner: DetectionStream (streaming の Step 1/2)。
        dispatcher: SkillDispatchLowerPolicy (skill 名 → Skill instance)。
        initial_skill: 初期 skill (default = "move_to_table"、Orchestrator の
            SkillState 初期値 + 初回 tick 時に dispatcher で auto start)。
        pick_leg_ref_n_avg_frames: base_rotation_start_table_top_verts の
            N-frame ring size (default 3、enter_pick の Kabsch/aspect 判定用)。
        transitions: 遷移グラフ (default = DEFAULT_TRANSITIONS)。
        enter_check: fire 判定関数 map (default = DEFAULT_ENTER_CHECK)。
        actuator_send_fn: Type B skill の action tensor を送信する callable。
            None なら silently drop (mock e2e / debug 時)。
        joint_state_source: Optional な joint state 供給 source。渡されると
            obs["joint_state"] に latest snapshot が乗る (未受信は None)。
        wrist_left_source / wrist_right_source: Optional な wrist camera source。
            渡されると obs["wrist_{side}_rgb"] に FrameData (未受信は None) が乗る。
            Mock skill は使わないが VLA drop-in で必要になる観測 pipeline の verify 用。
        head_perception_view: detectorへ渡すhead view。learned policyへ渡す
            `head_rgb`は常に元frameのまま保持する。
        log_sink: JSONL log 追記先 (TextIO)。None なら log 無し。
    """

    def __init__(
        self,
        perception: PerceptionProtocol,
        cleaner: DetectionStream,
        dispatcher: SkillDispatchLowerPolicy,
        initial_skill: str = "move_to_table",
        pick_leg_ref_n_avg_frames: int = 3,
        transitions: Optional[dict[str, list[str]]] = None,
        enter_check: Optional[
            dict[str, Callable[[list[OBBDetection], SkillState], bool]]
        ] = None,
        actuator_send_fn: Optional[Callable[[np.ndarray], None]] = None,
        joint_state_source: Optional[JointStateSourceProtocol] = None,
        dex1_state_source: Optional[Dex1StateSourceProtocol] = None,
        wrist_left_source: Optional[FrameSource] = None,
        wrist_right_source: Optional[FrameSource] = None,
        head_perception_view: str = "packed",
        log_sink: Optional[TextIO] = None,
        hard_timeout_by_skill: Optional[dict[str, float]] = None,
    ) -> None:
        if head_perception_view not in {"packed", "left", "right"}:
            raise ValueError(
                "head_perception_view must be one of 'packed', 'left', 'right', "
                f"got {head_perception_view!r}"
            )
        self.perception = perception
        self.cleaner = cleaner
        self.dispatcher = dispatcher
        self.state = SkillState(
            current_skill=initial_skill,
            n_legs_completed=0,
        )
        self._table_top_ring: deque[np.ndarray] = deque(
            maxlen=pick_leg_ref_n_avg_frames
        )
        self._last_frame_t: Optional[int] = None
        self.transitions = (
            transitions if transitions is not None else DEFAULT_TRANSITIONS
        )
        self.enter_check = (
            enter_check if enter_check is not None else DEFAULT_ENTER_CHECK
        )
        self.actuator_send_fn = actuator_send_fn
        self.joint_state_source = joint_state_source
        self.dex1_state_source = dex1_state_source
        self.wrist_left_source = wrist_left_source
        self.wrist_right_source = wrist_right_source
        self.head_perception_view = head_perception_view
        self.log_sink = log_sink
        self.hard_timeout_by_skill = dict(hard_timeout_by_skill or {})
        for skill_name, timeout_s in self.hard_timeout_by_skill.items():
            if timeout_s <= 0:
                raise ValueError(
                    f"hard timeout for {skill_name!r} must be > 0, got {timeout_s}"
                )

    def tick(self, frame: FrameData) -> Optional[TickResult]:
        """1 frame の pipeline。buffer 充填中 / 重複 t は None。

        初回 tick で dispatcher が initial_skill を auto start (二重 start 防止)。
        1 tick で最大 1 transition (for-break で enforce)。
        """
        # a) 重複 tick 防止 (source が同じ t を 2 度返した場合)
        if frame.t == self._last_frame_t:
            return None
        self._last_frame_t = frame.t

        # b) 初回 tick で initial skill を dispatcher で auto start
        if self.dispatcher.active_skill_name is None:
            self.dispatcher.start(
                self.state.current_skill,
                self._build_params(self.state.current_skill),
            )

        # c) YOLO → cleaner (streaming、buffer 充填中は None)
        # Keep packed stereo intact for learned policies.  Only the detector
        # receives the selected eye on which it was trained; slicing the source
        # itself would make VlaSkill duplicate HEAD_LEFT into HEAD_RIGHT.
        raw = self.perception.predict(self._perception_rgb(frame.rgb))
        cleaned = self.cleaner.push(raw)
        if cleaned is None:
            return None

        # d) table_top ring 更新 (transition ctx snapshot 用)
        top = _table_top_verts(cleaned)
        if top is not None:
            self._table_top_ring.append(top)

        # e) 履歴 flag latch + counter (SkillState.update に閉じる)
        self.state.update(cleaned)

        # f) fire dispatch (1 tick 1 transition、for-break で enforce)。
        # snapshot 用に transition 前 skill 名を確保しておく (使うのは fired 時のみ)。
        fired_to: Optional[str] = None
        prev_skill = self.state.current_skill
        for cand in self.transitions.get(self.state.current_skill, []):
            if self.enter_check[cand](cleaned, self.state):
                ctx = self._build_transition_ctx(cand)
                self.state.transition(cand, ctx)
                self.dispatcher.start(cand, self._build_params(cand))
                fired_to = cand
                break

        # f2) 遷移が起きた tick は stderr log 出力のみ (snapshot 経路は削除、Phase 3)。
        if fired_to is not None:
            print(
                f"[orch] fire transition: {prev_skill} -> {fired_to}",
                file=sys.stderr,
            )

        # g) skill inference (per-tick action)
        obs = self._build_obs(frame, cleaned)
        action = self.dispatcher.step(obs)
        if action is not None and self.actuator_send_fn is not None:
            self.actuator_send_fn(action)

        # h) result + JSONL log
        result = TickResult(
            t=frame.t,
            current_skill=self.state.current_skill,
            fire_transition_to=fired_to,
            cleaned=cleaned,
            action=action,
        )
        if self.log_sink is not None:
            self._log(result)
        return result

    def run(
        self, source: FrameSource, hz: Optional[float] = 30.0
    ) -> None:
        """FrameSource から pull で loop。ep 終端 (source.get() → None) で自動終了。

        Args:
            source: FrameSource (LerobotFrameSource / 将来 Ros2FrameSource)。
            hz: real-time cadence。None なら sleep 無し (ep replay 最速)、
                hz>0 なら 1/hz 秒 sleep (real-time)。
        """
        dt = 1.0 / hz if hz else 0.0
        while True:
            frame = source.get()
            if frame is None:
                break
            self.tick(frame)
            if dt > 0:
                time.sleep(dt)

    def run_live(
        self,
        source: FrameSource,
        *,
        hz: float = 30.0,
        startup_timeout: float = 10.0,
        frame_timeout: float = 1.0,
        stop_after_skill: Optional[str] = None,
        stop_after_s: Optional[float] = None,
        stop_on_skills: frozenset[str] = frozenset(),
        terminal_hard_timeout_returns: bool = False,
    ) -> None:
        """Live source を待機・freshness監視しながら実行する。

        ``run()`` の ``None`` は replay source の EOF を意味する。一方、live
        source は最初のDDS message受信前にも ``None`` を返すため、同じloopを
        使うと初回frameとのraceで即終了する。このmethodは最初のframeをtimeout
        付きで待ち、受信後はtimestampが更新されない状態も安全異常として扱う。

        各 skill の最大 dwell 秒は ``Skill.max_dwell_sec`` property で
        skill 自身が持ち、Orchestrator は dispatcher.active_skill 経由で lookup
        する (Issue #81)。指定秒を超えて active のままなら TRANSITIONS graph の
        first candidate に **auto-transition** で強制遷移し pipeline を継続する
        (move_to_table のように enter_check が _enter_never で dwell timer 経由で
        のみ進む skill の正規経路、および他 skill で YOLO fire が来ない domain gap
        時のフォールバック、両方を兼ねる)。実機actuatorではSDK command自体にも同じ
        有限durationを設定し、process hang時にもfirmware側で速度指令が失効する
        構成を前提とする。
        fallback: 遷移候補ゼロなら ``LiveSourceSafetyError`` を raise (最終防衛)。
        auto-transition は各活性化 1 回まで、skill が再度 active になった場合
        (state machine 内 cycle) は再度 fire する。
        """
        if hz <= 0:
            raise ValueError(f"hz must be > 0 for live source, got {hz}")
        if startup_timeout <= 0:
            raise ValueError(
                f"startup_timeout must be > 0, got {startup_timeout}"
            )
        if frame_timeout <= 0:
            raise ValueError(f"frame_timeout must be > 0, got {frame_timeout}")
        if (stop_after_skill is None) != (stop_after_s is None):
            raise ValueError(
                "stop_after_skill and stop_after_s must be specified together"
            )
        if stop_after_s is not None and stop_after_s <= 0:
            raise ValueError(f"stop_after_s must be > 0, got {stop_after_s}")

        dt = 1.0 / hz
        wait_started_at = time.monotonic()
        last_fresh_frame_at: Optional[float] = None
        # active skill の chain-aware dwell 追跡: skill 名が変化したら
        # started_at をリセットし、dwell_fired は False に戻る。
        active_skill_name: Optional[str] = None
        active_skill_started_at: Optional[float] = None
        active_skill_dwell_fired: bool = False

        while True:
            frame = source.get()
            now = time.monotonic()

            if frame is None:
                if last_fresh_frame_at is None and (
                    now - wait_started_at >= startup_timeout
                ):
                    raise LiveSourceSafetyError(
                        "camera startup timeout: no frame received within "
                        f"{startup_timeout:g}s"
                    )
                if last_fresh_frame_at is not None and (
                    now - last_fresh_frame_at >= frame_timeout
                ):
                    raise LiveSourceSafetyError(
                        "camera frame timeout: no frame available for "
                        f"{frame_timeout:g}s"
                    )
            elif frame.t != self._last_frame_t:
                self.tick(frame)
                now = time.monotonic()
                last_fresh_frame_at = now
            elif (
                last_fresh_frame_at is not None
                and now - last_fresh_frame_at >= frame_timeout
            ):
                raise LiveSourceSafetyError(
                    "camera frame timeout: timestamp did not advance for "
                    f"{frame_timeout:g}s"
                )

            # active skill 変化検出 → dwell timer リセット。
            # tick 内の fire dispatch や auto-transition 経由で
            # dispatcher.active_skill_name が変わった時にここで捕捉する。
            active_skill_obj = self.dispatcher.active_skill
            current_active = active_skill_obj.name if active_skill_obj is not None else None
            if current_active != active_skill_name:
                active_skill_name = current_active
                active_skill_started_at = now if current_active is not None else None
                active_skill_dwell_fired = False
                if current_active in stop_on_skills:
                    print(
                        f"[orch] operator stop boundary reached: transition -> "
                        f"{current_active}; stopping before any later skill",
                        file=sys.stderr,
                    )
                    return

            if (
                current_active == stop_after_skill
                and active_skill_started_at is not None
                and stop_after_s is not None
                and now - active_skill_started_at >= stop_after_s
            ):
                print(
                    f"[orch] operator stop boundary reached: {current_active} ran "
                    f"{stop_after_s:g}s; stopping",
                    file=sys.stderr,
                )
                return

            # Observation-driven finite skills (notably the collision-aware
            # post-walk arm path) transition only after their measured target
            # converges.  This must run before max_dwell handling so a timeout
            # can never be mistaken for successful completion.
            if (
                active_skill_obj is not None
                and active_skill_obj.is_complete
                and not active_skill_dwell_fired
            ):
                candidates = self.transitions.get(current_active, [])
                if not candidates:
                    print(
                        f"[orch] terminal skill {current_active!r} completed; "
                        "stage finished",
                        file=sys.stderr,
                    )
                    return
                cand = candidates[0]
                ctx = self._build_transition_ctx(cand)
                self.state.transition(cand, ctx)
                self.dispatcher.start(cand, self._build_params(cand))
                print(
                    f"[orch] skill_complete transition: {current_active} -> {cand}",
                    file=sys.stderr,
                )
                active_skill_dwell_fired = True

            # Dataset-derived learned-skill timeout is a fail-closed boundary,
            # not evidence that the manipulation succeeded.  Never advance to
            # the next physical skill merely because this timer expired.
            hard_timeout = self.hard_timeout_by_skill.get(current_active or "")
            if (
                hard_timeout is not None
                and active_skill_started_at is not None
                and now - active_skill_started_at >= hard_timeout
            ):
                if (
                    terminal_hard_timeout_returns
                    and not self.transitions.get(current_active, [])
                ):
                    print(
                        f"[orch] terminal skill {current_active!r} reached YAML "
                        f"max_seconds_hard={hard_timeout:g}s; holding at the "
                        "operator stage boundary",
                        file=sys.stderr,
                    )
                    return
                raise LiveSourceSafetyError(
                    f"skill {current_active!r} reached YAML max_seconds_hard="
                    f"{hard_timeout:g}s; holding and stopping this stage without "
                    "transition"
                )

            # active skill が自身の max_dwell_sec を expose していれば dwell 判定。
            # None expose なら enter_check ベースのみ (fail-safe 無し)。
            if (
                active_skill_obj is not None
                and not active_skill_dwell_fired
                and active_skill_started_at is not None
                and active_skill_obj.max_dwell_sec is not None
            ):
                max_dwell = active_skill_obj.max_dwell_sec
                elapsed = now - active_skill_started_at
                if elapsed >= max_dwell:
                    candidates = self.transitions.get(current_active, [])
                    if candidates:
                        cand = candidates[0]
                        ctx = self._build_transition_ctx(cand)
                        self.state.transition(cand, ctx)
                        self.dispatcher.start(cand, self._build_params(cand))
                        print(
                            f"[orch] skill_max_dwell ({max_dwell:g}s) "
                            f"auto-transition: {current_active} -> {cand}",
                            file=sys.stderr,
                        )
                        # この活性化ぶんは発火済にして再発火防止。次 iteration で
                        # dispatcher.active_skill_name の変化を検出して timer リセット。
                        active_skill_dwell_fired = True
                    else:
                        raise LiveSourceSafetyError(
                            f"skill {current_active!r} exceeded "
                            f"{max_dwell:g}s and no transition candidates"
                        )

            time.sleep(dt)

    # ---- 内部 helpers ----
    def _build_transition_ctx(self, next_skill: str) -> Optional[dict]:
        """transition() に渡す ctx。move_table_base に遷移する時は table_top verts
        の N-frame pivot-aligned mean を snapshot として渡す。"""
        if next_skill == "move_table_base" and self._table_top_ring:
            return {
                "base_rotation_start_table_top_verts": mean_verts_pivot_aligned(
                    list(self._table_top_ring)
                )
            }
        return None

    def _build_params(self, skill: str) -> dict:
        """dispatcher.start(skill, params) に渡す params。

        現状 MockSkill は無視する (空 dict)。将来 Gr00tSkill が task_prompt を
        受ける形になったら、skill 名 → prompt map を hook として注入する予定。
        """
        return {}

    def _build_obs(
        self, frame: FrameData, cleaned: list[OBBDetection]
    ) -> dict:
        """dispatcher.step(obs) に渡す観測 dict。

        head_rgb + t + cleaned は常に含む。以下 Optional source が渡されていれば
        obs にも field 追加 (source.get() が None = 未受信でも field は追加、値 None):
          - joint_state_source → `joint_state` (JointStateData or None)
          - wrist_left_source  → `wrist_left_rgb` (FrameData or None)
          - wrist_right_source → `wrist_right_rgb` (FrameData or None)

        SampleVLASkill (Mock) は obs 使わない (固定 pose) が、real VLA drop-in で
        step() 内 `self._model.predict(obs)` に差し替えれば、head + wrist RGB +
        joint state の 4 modality 揃った input が使える構造。
        """
        obs: dict = {"head_rgb": frame.rgb, "t": frame.t, "cleaned": cleaned}
        if self.joint_state_source is not None:
            obs["joint_state"] = self.joint_state_source.get()
        if self.dex1_state_source is not None:
            obs["hand_state"] = self.dex1_state_source.get()
        if self.wrist_left_source is not None:
            obs["wrist_left_rgb"] = self.wrist_left_source.get()
        if self.wrist_right_source is not None:
            obs["wrist_right_rgb"] = self.wrist_right_source.get()
        return obs

    def _perception_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Select a detector eye without changing the policy observation."""
        if self.head_perception_view == "packed":
            return rgb
        width = int(rgb.shape[1])
        if width < 2 or width % 2 != 0:
            raise ValueError(
                "packed stereo image must have a positive even width when "
                f"head_perception_view={self.head_perception_view!r}, got {width}"
            )
        split = width // 2
        return (
            rgb[:, :split]
            if self.head_perception_view == "left"
            else rgb[:, split:]
        )

    def _log(self, result: TickResult) -> None:
        """JSONL 1 行を log_sink に追記。detection の詳細は書かず summary のみ
        (詳細 debug は別 script)。"""
        payload = {
            "t": result.t,
            "current_skill": result.current_skill,
            "fire_transition_to": result.fire_transition_to,
            "n_cleaned": len(result.cleaned),
            "action_shape": (
                list(result.action.shape)
                if result.action is not None
                else None
            ),
        }
        self.log_sink.write(json.dumps(payload) + "\n")  # type: ignore[union-attr]

    # Resume mechanism (restore_from_snapshot / _emit_transition_snapshot) は
    # Issue #128 Phase 3 完了に伴い削除。--stage=N で再実行が中断復旧の代替。
