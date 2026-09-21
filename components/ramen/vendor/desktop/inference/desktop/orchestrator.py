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
from typing import Any, Callable, Optional, Protocol, TextIO

import numpy as np

from inference.desktop.assembly import (
    CAMERA_STALE_TIMEOUT_S,
    build_observation,
    camera_stale_roles,
    head_procedure_skill_names,
    select_head_view,
)
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
# skill が時間切れになったときの動き (Issue #141 束 1-13 / D4)。
TIMEOUT_ACTIONS = ("advance", "stop")

DEFAULT_TRANSITIONS: dict[str, list[str]] = {
    "setup": ["move_to_table"],  # Issue #81 Phase 3: 腕 pre-motion
    "move_to_table": ["move_table_base"],
    "move_table_base": ["pick_table_leg"],
    "pick_table_leg": ["insert_table_leg"],
    "insert_table_leg": ["rotate_leg_to_tighten"],
    "rotate_leg_to_tighten": ["flip_table", "move_table_base"],  # ★ priority = list 順
    "flip_table": [],
}


def enter_never(dets: list[OBBDetection], state: SkillState) -> bool:
    """timer (max_dwell_sec) のみで進む skill 用の enter_check。YOLO 検出では fire しない。

    setup / move_to_table は前段の完了 (=時間) で次段へ進むため、enter_check では常に
    False を返す。tick() は enter_check[cand] を直接 index するので、dwell-only の遷移先も
    registry に entry が要る (未登録だと KeyError で即死する。Issue #97)。
    """
    return False


DEFAULT_ENTER_CHECK: dict[str, Callable[[list[OBBDetection], SkillState], bool]] = {
    "move_to_table": enter_never,  # dwell (max_dwell_sec) のみで move_table_base へ
    "move_table_base": enter_move_table_base,
    "pick_table_leg": enter_pick_table_leg,
    "insert_table_leg": enter_insert_table_leg,
    "rotate_leg_to_tighten": enter_rotate_leg_to_tighten,
    "flip_table": enter_flip_table,
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
    # Stage 0 ends with the arms still in the measured lowered pose, then runs the
    # head procedure for the first pick.  Issue #141 (D7-1) moved the per-skill
    # pre-motion into that head procedure, so these lists hold only the skills that
    # are specific to the stage itself.
    0: ["setup", "move_to_table", "post_walk_settle"],
    1: ["pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"],
    2: [
        "rotate_table_base",
        "pick_table_leg",
        "insert_table_leg",
        "rotate_leg_to_tighten",
    ],
    3: [
        "rotate_table_base",
        "pick_table_leg",
        "insert_table_leg",
        "rotate_leg_to_tighten",
    ],
    4: [
        "rotate_table_base",
        "pick_table_leg",
        "insert_table_leg",
        "rotate_leg_to_tighten",
    ],
    5: ["flip_table"],
}
# どの skill の開始姿勢へ向けて頭の手順を入れるか (Issue #141 D2 / D7-1)。
# stage 0 は最後に (歩いた後に) 入れる。stage 1 は stage 0 で済んでいるので、
# stage 1 から起動したときだけ先頭に入れる。
STAGE_HEAD_SKILL: dict[int, str] = {
    0: "pick_table_leg",
    1: "pick_table_leg",
    2: "rotate_table_base",
    3: "rotate_table_base",
    4: "rotate_table_base",
    5: "flip_table",
}
STAGE_MIN: int = 0
STAGE_MAX: int = 5


def build_stage_transitions(
    stage: int, *, is_start_stage: bool = True, include_hand: bool = True
) -> dict[str, list[str]]:
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
    skills = build_stage_skill_sequence(
        stage, is_start_stage=is_start_stage, include_hand=include_hand
    )
    transitions: dict[str, list[str]] = {}
    for i, skill in enumerate(skills):
        transitions[skill] = [skills[i + 1]] if i + 1 < len(skills) else []
    return transitions


def build_stage_skill_sequence(
    stage: int, *, is_start_stage: bool = True, include_hand: bool = True
) -> list[str]:
    """Stage で dispatch する skill 名の列 (頭の手順を含む、Issue #141 D7-1)。

    頭の手順 (手を開く → 腕の pre-motion → 手を開始の開度へ → N 秒保持) は、
    次に始める学習 skill の開始姿勢へ体を持っていく手順。

    - stage 0: 歩いた後に、1 本目の pick 用の頭の手順を行う (列の最後)
    - stage 1: stage 0 で済んでいるので入れない。**stage 1 から起動したときだけ**先頭に入れる
    - stage 2〜5: 前の脚の締め終わりから始まるので、必ず先頭に入れる

    Args:
        stage: STAGE_SKILL_SEQUENCES の key (0..5)。
        is_start_stage: この stage から起動したか (途中から再開したか)。
        include_hand: Dex1 を実際に動かすか。False なら手の skill を列に入れない。
    """
    base = list(STAGE_SKILL_SEQUENCES[stage])
    head_skill = STAGE_HEAD_SKILL.get(stage)
    if stage == 1 and not is_start_stage:
        head_skill = None
    if head_skill is None:
        return base
    head = head_procedure_skill_names(head_skill, include_hand=include_hand)
    return base + head if stage == 0 else head + base


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
    """1 tick の出力 (debug / test / log 用)。

    Attributes:
        cleaned: 上位 Planner に渡した検出 (`DetectionStream`)。溜まっていない tick と、
            新しい frame が来ていない tick は None。
        policy_cleaned: policy (overlay / memory / progress monitor) に渡した検出。
            遅れの無い filter を通したもの。検出がまだ 1 度も無ければ None。
        detection_refreshed: この tick で新しい frame に YOLO を掛けたか。
    """

    t: int
    current_skill: str
    fire_transition_to: Optional[str]  # transition 起きたら新 skill 名、無ければ None
    cleaned: Optional[list[OBBDetection]]
    action: Optional[np.ndarray]  # dispatcher.step の返り値 (Type B の action tensor)
    policy_cleaned: Optional[list[OBBDetection]] = None
    detection_refreshed: bool = True


@dataclass
class SkillAdvance:
    """`advance_finished_skill()` の結果。

    Attributes:
        fired_to: 遷移した先の skill 名。遷移しなければ None。
        reason: "complete" / "timeout" / "dwell" のどれで進んだか。
        stage_finished: 終端 skill が完了 or 時間切れした。呼び出し側が
            stage を畳む (`run_live` は return、boundary は以後進めない)。
    """

    fired_to: Optional[str] = None
    reason: Optional[str] = None
    stage_finished: bool = False


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
        policy_filter: Optional[Any] = None,
        timeout_action_by_skill: Optional[dict[str, str]] = None,
        camera_stale_timeout_s: float = CAMERA_STALE_TIMEOUT_S,
        on_tick: Optional[Callable[["TickResult", dict, Any], None]] = None,
    ) -> None:
        if head_perception_view not in {"packed", "left", "right"}:
            raise ValueError(
                "head_perception_view must be one of 'packed', 'left', 'right', "
                f"got {head_perception_view!r}"
            )
        self.perception = perception
        self.cleaner = cleaner
        self.dispatcher = dispatcher
        # `reset_episode()` が最初の skill に戻すために覚えておく。
        self._initial_skill = initial_skill
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
        # Issue #141 D3: policy に渡す検出は遅れの無い filter を通す (Planner は cleaner のまま)。
        # None なら cleaner の出力をそのまま policy にも渡す (従来動作)。
        self.policy_filter = policy_filter
        self.camera_stale_timeout_s = float(camera_stale_timeout_s)
        if self.camera_stale_timeout_s <= 0:
            raise ValueError(
                f"camera_stale_timeout_s must be > 0, got {camera_stale_timeout_s}"
            )
        self.on_tick = on_tick
        self._policy_cleaned: Optional[list[OBBDetection]] = None
        # active skill の経過時間追跡。`advance_finished_skill()` が使う。
        # run_live() の局所変数だったものを instance に上げた: boundary の driver は
        # tick() を直接回すので、局所変数のままだと **大会経路だけ受け皿を失う**。
        self._active_skill_name: Optional[str] = None
        self._active_skill_started_at: Optional[float] = None
        self._active_skill_dwell_fired = False
        self.hard_timeout_by_skill = dict(hard_timeout_by_skill or {})
        # Issue #141 D4: 時間切れの動き。既定は「次の skill へ進む」。
        self.timeout_action_by_skill = dict(timeout_action_by_skill or {})
        for skill_name, action in self.timeout_action_by_skill.items():
            if action not in TIMEOUT_ACTIONS:
                raise ValueError(
                    f"on_timeout for {skill_name!r} must be one of {TIMEOUT_ACTIONS}, "
                    f"got {action!r}"
                )
        for skill_name, timeout_s in self.hard_timeout_by_skill.items():
            if timeout_s <= 0:
                raise ValueError(
                    f"hard timeout for {skill_name!r} must be > 0, got {timeout_s}"
                )

    def tick(self, frame: FrameData) -> TickResult:
        """1 tick の pipeline。

        新しい frame が来た tick だけ YOLO を掛け、上位 Planner の判定を行う。
        cleaner が溜まっていない tick も、同じ frame で 2 度目の tick も、skill は
        進める (Issue #141 D3 / D4)。初回 tick で dispatcher が initial_skill を
        auto start する。1 tick で最大 1 transition。
        """
        # a) 初回 tick で initial skill を dispatcher で auto start
        if self.dispatcher.active_skill_name is None:
            self.dispatcher.start(
                self.state.current_skill,
                self._build_params(self.state.current_skill),
            )

        # b) YOLO は新しい frame の tick だけ (Issue #141 D4)。同じ frame を filter に
        #    2 度入れると「前の frame にも出ていた」の判定が崩れる。新しい frame が
        #    来ていない tick は、前の検出のまま skill だけ進める。
        detection_refreshed = frame.t != self._last_frame_t
        planner_cleaned: Optional[list[OBBDetection]] = None
        if detection_refreshed:
            self._last_frame_t = frame.t
            # Keep packed stereo intact for learned policies.  Only the detector
            # receives the selected eye on which it was trained.
            raw = self.perception.predict(self._perception_rgb(frame.rgb))
            planner_cleaned = self.cleaner.push(raw)
            self._policy_cleaned = (
                self.policy_filter.push(raw)
                if self.policy_filter is not None
                else planner_cleaned
            )

        # c) 上位 Planner の判定は、cleaner の出力が出た tick だけ。溜まっていない間も
        #    skill は止めない (Issue #141 D3)。
        fired_to: Optional[str] = None
        if planner_cleaned is not None:
            top = _table_top_verts(planner_cleaned)
            if top is not None:
                self._table_top_ring.append(top)
            self._seed_base_rotation_reference_if_needed()
            self.state.update(planner_cleaned)
            prev_skill = self.state.current_skill
            for cand in self.transitions.get(self.state.current_skill, []):
                if self.enter_check[cand](planner_cleaned, self.state):
                    ctx = self._build_transition_ctx(cand)
                    self.state.transition(cand, ctx)
                    self.dispatcher.start(cand, self._build_params(cand))
                    fired_to = cand
                    break
            if fired_to is not None:
                print(
                    f"[orch] fire transition: {prev_skill} -> {fired_to}",
                    file=sys.stderr,
                )

        # d) 観測 → skill (評価経路と同じ組み立て、Issue #141 D3)
        obs = self._build_obs(frame, self._policy_cleaned)
        self._check_camera_freshness(obs)
        action = self.dispatcher.step(obs)
        if action is not None and self.actuator_send_fn is not None:
            self.actuator_send_fn(action)

        # e) result + JSONL log + 記録の hook
        result = TickResult(
            t=frame.t,
            current_skill=self.state.current_skill,
            fire_transition_to=fired_to,
            cleaned=planner_cleaned,
            action=action,
            policy_cleaned=self._policy_cleaned,
            detection_refreshed=detection_refreshed,
        )
        if self.log_sink is not None:
            self._log(result)
        if self.on_tick is not None:
            self.on_tick(result, obs, self.dispatcher.active_skill)
        return result

    def _check_camera_freshness(self, obs: dict) -> None:
        """4 台のカメラの受信時刻を見て、止まっていれば安全停止に回す (Issue #141 D3)。

        画像は latest-only なので、USB が抜けても最後の画像が入り続ける。頭だけでなく
        手首も同じ基準で見る (学習と違う入力のまま動かさない)。
        """
        require_wrist = (
            self.wrist_left_source is not None and self.wrist_right_source is not None
        )
        stale = camera_stale_roles(
            obs, require_wrist=require_wrist, max_age_s=self.camera_stale_timeout_s
        )
        if stale:
            detail = ", ".join(
                f"{role}={age:.3f}s" for role, age in sorted(stale.items())
            )
            raise LiveSourceSafetyError(
                f"camera frames are stale (> {self.camera_stale_timeout_s:g}s): {detail}"
            )

    def run(self, source: FrameSource, hz: Optional[float] = 30.0) -> None:
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
        (move_to_table のように enter_check が enter_never で dwell timer 経由で
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
            raise ValueError(f"startup_timeout must be > 0, got {startup_timeout}")
        if frame_timeout <= 0:
            raise ValueError(f"frame_timeout must be > 0, got {frame_timeout}")
        if (stop_after_skill is None) != (stop_after_s is None):
            raise ValueError(
                "stop_after_skill and stop_after_s must be specified together"
            )
        if stop_after_s is not None and stop_after_s <= 0:
            raise ValueError(f"stop_after_s must be > 0, got {stop_after_s}")

        dt = 1.0 / hz
        # 次の締め切りまで寝る (処理時間を含めて 30 Hz)。遅れた分は取り戻さない
        # (短い間隔で指令を出すと、教師より速い動きになる)。Issue #141 D4。
        next_deadline = time.monotonic()
        wait_started_at = time.monotonic()
        last_fresh_frame_at: Optional[float] = None
        # 区切り (stop_on_skills) 用に「どの skill まで判定したか」を別に持つ。
        # dwell の追跡とは別にしないと、tick の中で起きた遷移を取りこぼす。
        boundary_seen: Optional[str] = None

        while True:
            # 区切りの判定は **tick の前**。ここで戻ることで、次の skill を 1 tick も
            # 動かさずに操作者の Enter を待てる (Issue #141 D6-1)。
            active_skill_obj = self.dispatcher.active_skill
            current_active = (
                active_skill_obj.name if active_skill_obj is not None else None
            )
            if current_active != boundary_seen:
                boundary_seen = current_active
                if current_active in stop_on_skills:
                    print(
                        f"[orch] operator stop boundary reached: transition -> "
                        f"{current_active}; stopping before any later skill",
                        file=sys.stderr,
                    )
                    return

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
            else:
                # 新しい frame が来ていなくても skill は 30 Hz で進める。YOLO と
                # filter は新しい frame の tick だけ (tick 側で判定、Issue #141 D4)。
                is_new_frame = frame.t != self._last_frame_t
                if (
                    not is_new_frame
                    and last_fresh_frame_at is not None
                    and now - last_fresh_frame_at >= frame_timeout
                ):
                    raise LiveSourceSafetyError(
                        "camera frame timeout: timestamp did not advance for "
                        f"{frame_timeout:g}s"
                    )
                self.tick(frame)
                now = time.monotonic()
                if is_new_frame:
                    last_fresh_frame_at = now

            # active skill の変化検出 / 経過時間追跡は advance_finished_skill() が持つ。
            active_skill_obj = self.dispatcher.active_skill
            current_active = (
                active_skill_obj.name if active_skill_obj is not None else None
            )

            # Finite rule-based skills retain their last safe target when an
            # IK/grasp/stage check fails.  Surface that latched failure here so
            # the real runner performs its controlled arm release instead of
            # silently holding until an unrelated dwell timeout.
            failure_reason = (
                getattr(active_skill_obj, "failure_reason", None)
                if active_skill_obj is not None
                else None
            )
            if failure_reason:
                raise LiveSourceSafetyError(
                    f"skill {current_active!r} failed while holding its last "
                    f"safe target: {failure_reason}"
                )

            # is_complete / max_seconds_hard / max_dwell_sec。boundary の driver も
            # 同じ method を呼ぶ (片方にしか無いと大会経路だけ受け皿を失う)。
            advance = self.advance_finished_skill(now)
            if advance.stage_finished:
                return

            # operator の stop 境界は advance より後。時計は advance 側が持つ。
            started_at = self._active_skill_started_at
            if (
                self._active_skill_name == stop_after_skill
                and started_at is not None
                and stop_after_s is not None
                and now - started_at >= stop_after_s
            ):
                print(
                    f"[orch] operator stop boundary reached: {self._active_skill_name} "
                    f"ran {stop_after_s:g}s; stopping",
                    file=sys.stderr,
                )
                return

            next_deadline += dt
            sleep_s = next_deadline - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_deadline = time.monotonic() + dt

    def advance_finished_skill(self, now: Optional[float] = None) -> SkillAdvance:
        """active skill が終わったか時間切れなら次へ進める。1 回の呼び出しで最大 1 遷移。

        `tick()` は **YOLO の `enter_check` しか見ない**。この method はその受け皿で、
        3 つの signal を順に見る:

          1. `is_complete`      skill が自分で「収束した」と言う (腕の pre-motion 等)
          2. `max_seconds_hard` YAML の時間切れ。`on_timeout` は advance / stop
          3. `max_dwell_sec`    skill が expose していれば dwell 上限

        ⚠️ **`run_live()` と boundary の `OrchestratorDriver` の両方から呼ぶこと。**
        元は `run_live()` の局所変数で閉じていたため、`tick()` を直接回す大会経路には
        受け皿が無く、YOLO が落とすと `rotate_table_base` から永久に出られなかった
        (2026-09-21)。片方にしか無いと同じことが再発する。

        時間切れは「うまくいった証拠ではない」ので、進んだ理由は必ず stderr に残す。

        Args:
            now: 単調時刻。省略時は `time.monotonic()`。

        Returns:
            SkillAdvance。遷移しなければ `fired_to is None`。

        Raises:
            LiveSourceSafetyError: `on_timeout: stop` の skill が時間切れした、
                または dwell 上限に達したのに遷移先が無い。
        """
        now = time.monotonic() if now is None else now
        active_skill_obj = self.dispatcher.active_skill
        current_active = (
            active_skill_obj.name if active_skill_obj is not None else None
        )
        # active skill が変わったら timer を張り直す。tick() 内の enter_check 発火でも
        # 変わるので、呼び出し側ではなくここで検出する。
        if current_active != self._active_skill_name:
            self._active_skill_name = current_active
            self._active_skill_started_at = now if current_active is not None else None
            self._active_skill_dwell_fired = False

        started_at = self._active_skill_started_at

        def _advance(cand: str, reason: str, message: str) -> SkillAdvance:
            ctx = self._build_transition_ctx(cand)
            self.state.transition(cand, ctx)
            self.dispatcher.start(cand, self._build_params(cand))
            print(message, file=sys.stderr)
            self._active_skill_dwell_fired = True
            return SkillAdvance(fired_to=cand, reason=reason)

        # 1) 観測駆動の有限 skill (腕の pre-motion 等) は measured target が収束して
        #    初めて完了する。**max_dwell より先に見る**ので、時間切れを「成功」と
        #    取り違えない。
        if (
            active_skill_obj is not None
            and active_skill_obj.is_complete
            and not self._active_skill_dwell_fired
        ):
            candidates = self.transitions.get(current_active, [])
            if not candidates:
                print(
                    f"[orch] terminal skill {current_active!r} completed; "
                    "stage finished",
                    file=sys.stderr,
                )
                return SkillAdvance(reason="complete", stage_finished=True)
            return _advance(
                candidates[0],
                "complete",
                f"[orch] skill_complete transition: {current_active} -> {candidates[0]}",
            )

        # 2) 学習 skill の時間切れ (Issue #141 束 1-13 / D4)。
        hard_timeout = self.hard_timeout_by_skill.get(current_active or "")
        if (
            hard_timeout is not None
            and started_at is not None
            and now - started_at >= hard_timeout
        ):
            timeout_action = self.timeout_action_by_skill.get(
                current_active or "", "advance"
            )
            if timeout_action == "stop":
                raise LiveSourceSafetyError(
                    f"skill {current_active!r} reached YAML max_seconds_hard="
                    f"{hard_timeout:g}s; holding and stopping this stage without "
                    "transition"
                )
            candidates = self.transitions.get(current_active, [])
            if not candidates:
                print(
                    f"[orch] terminal skill {current_active!r} reached YAML "
                    f"max_seconds_hard={hard_timeout:g}s; stage finished",
                    file=sys.stderr,
                )
                return SkillAdvance(reason="timeout", stage_finished=True)
            if not self._active_skill_dwell_fired:
                return _advance(
                    candidates[0],
                    "timeout",
                    f"[orch] skill_timeout ({hard_timeout:g}s) advance: "
                    f"{current_active} -> {candidates[0]}",
                )

        # 3) skill が max_dwell_sec を expose していれば dwell 判定。
        #    None expose なら enter_check ベースのみ (fail-safe 無し)。
        if (
            active_skill_obj is not None
            and not self._active_skill_dwell_fired
            and started_at is not None
            and active_skill_obj.max_dwell_sec is not None
        ):
            max_dwell = active_skill_obj.max_dwell_sec
            if now - started_at >= max_dwell:
                candidates = self.transitions.get(current_active, [])
                if not candidates:
                    raise LiveSourceSafetyError(
                        f"skill {current_active!r} exceeded "
                        f"{max_dwell:g}s and no transition candidates"
                    )
                return _advance(
                    candidates[0],
                    "dwell",
                    f"[orch] skill_max_dwell ({max_dwell:g}s) "
                    f"auto-transition: {current_active} -> {candidates[0]}",
                )

        return SkillAdvance()

    def reset_episode(self) -> None:
        """次の episode を最初からやり直せる状態に戻す。

        ⚠️ `dispatcher.stop()` は `VlaSkill._on_stop()` を通るので、**その時点で
        active だった skill の model は解放される** (skill の buffer を畳むのに
        必要な経路で、避けられない)。常駐から外れていない model は
        `ModelResidency` が background で読み直すので、次の episode の頭で
        止まるのは「reset 時に active だった skill が次も先頭に来る」場合だけ。

        boundary の server は 1 本の process が動き続け、運営が episode 間に
        `reset` を呼ぶ (`components/transport.py` の route)。ここを戻さないと
        2 本目が **skill を一切進めないまま終わる**:
          - `n_legs_completed` が 4 のままだと `enter_pick_table_leg` が常に False
          - `base_rotation_start_table_top_verts` が残ると 1 本目の天板が基準になる
          - `dispatcher` が前の skill を握ったままだと initial_skill に戻らない

        自前経路 (`run_live`) は stage ごとに `Orchestrator` を作り直すので
        呼ぶ必要は無い (呼んでも害は無い)。
        """
        self.dispatcher.stop()
        self.state = SkillState(
            current_skill=self._initial_skill,
            n_legs_completed=0,
        )
        self._table_top_ring.clear()
        self._last_frame_t = None
        self._policy_cleaned = None
        self._active_skill_name = None
        self._active_skill_started_at = None
        self._active_skill_dwell_fired = False
        print("[orch] episode reset", file=sys.stderr)

    # ---- 内部 helpers ----
    def _seed_base_rotation_reference_if_needed(self) -> None:
        """初期 skill が天板を回す skill のとき、基準を 1 度だけ seed する。

        `enter_pick_table_leg` は `state.base_rotation_start_table_top_verts` が
        `None` なら**必ず False を返す**。この基準は `state.transition()` の ctx
        経由でしか入らず、それを作る `_build_transition_ctx` は遷移のときしか
        走らない。

        ところが `rotate_table_base` が **initial_skill** の場合、tick() 冒頭の
        auto start は `dispatcher.start()` を直接呼ぶので `state.transition()` を
        通らない。結果、基準が一度も入らず **orchestrator は rotate_table_base
        から永久に出られない**。

        これは大会経路そのもの:
          - boundary の `OrchestratorDriver` は `initial_skill="rotate_table_base"`
          - 自前経路の `--stage 1`〜`4` も先頭 skill が `rotate_table_base`
        実 image を載せた pod で、実フレームを 57.5° 回しても
        (threshold は 18°) 遷移が 0 回であることを確認した (2026-09-21)。

        遷移経由の基準は「天板を回し始める直前の姿勢」なので、initial の場合の
        等価物は「この run が始まった時点の姿勢」。ring が満ちた最初の tick で
        取る (`pick_leg_ref_n_avg_frames` frame の平均、30Hz なら 0.1 秒以内)。
        """
        if self.state.base_rotation_start_table_top_verts is not None:
            return
        if self.state.current_skill not in ("move_table_base", "rotate_table_base"):
            return
        if len(self._table_top_ring) < (self._table_top_ring.maxlen or 1):
            return
        self.state.base_rotation_start_table_top_verts = mean_verts_pivot_aligned(
            list(self._table_top_ring)
        )
        print(
            "[orch] seeded base rotation reference for the initial skill "
            f"{self.state.current_skill!r} "
            f"({len(self._table_top_ring)} frame の平均)",
            file=sys.stderr,
        )

    def _build_transition_ctx(self, next_skill: str) -> Optional[dict]:
        """transition() に渡す ctx。天板を回す skill に入るとき、その直前の table_top
        verts の N-frame pivot-aligned mean を基準として渡す。

        `enter_pick_table_leg` はこの基準と今の天板を比べて rotate → pick を判定する。
        stage の流れには `move_table_base` が無い (rotate_table_base に分割した) ので、
        rotate に入るときも記録しないと基準が一度も入らず、pick へ永久に進まない
        (Issue #141 束 1-6 / INF-7)。
        """
        if (
            next_skill in ("move_table_base", "rotate_table_base")
            and self._table_top_ring
        ):
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
        self, frame: FrameData, cleaned: Optional[list[OBBDetection]]
    ) -> dict:
        """dispatcher.step(obs) に渡す観測。組み立ては評価経路と同じ関数を使う。

        `obs["t"]` はホストの monotonic ns、カメラの header.stamp と受信時刻は
        `_camera_generations` / `_camera_received_monotonic_ns` に入る
        (詳細は `assembly.build_observation`)。
        """
        return build_observation(
            head=frame,
            detections=cleaned,
            joint_state_source=self.joint_state_source,
            dex1_state_source=self.dex1_state_source,
            wrist_left_source=self.wrist_left_source,
            wrist_right_source=self.wrist_right_source,
        )

    def _perception_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """検出に渡す head の目を選ぶ (policy 用の観測は packed のまま)。"""
        return select_head_view(rgb, self.head_perception_view)

    def _log(self, result: TickResult) -> None:
        """JSONL 1 行を log_sink に追記。detection の詳細は書かず summary のみ
        (詳細 debug は別 script)。"""
        payload = {
            "t": result.t,
            "current_skill": result.current_skill,
            "fire_transition_to": result.fire_transition_to,
            "n_cleaned": None if result.cleaned is None else len(result.cleaned),
            "n_policy_cleaned": (
                None if result.policy_cleaned is None else len(result.policy_cleaned)
            ),
            "action_shape": (
                list(result.action.shape) if result.action is not None else None
            ),
        }
        self.log_sink.write(json.dumps(payload) + "\n")  # type: ignore[union-attr]

    # Resume mechanism (restore_from_snapshot / _emit_transition_snapshot) は
    # Issue #128 Phase 3 完了に伴い削除。--stage=N で再実行が中断復旧の代替。
