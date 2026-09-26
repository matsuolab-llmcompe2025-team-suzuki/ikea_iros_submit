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

# 関節 state が止まったと判断する受信からの経過 [s]。受信時刻を持つ source
# (公式 boundary の `:5557`、50 Hz) だけが対象。止まっても `get()` は最後の値を
# 返し続け、VlaSkill も前 tick の値で組み続けるので、ここで止める。
JOINT_STATE_STALE_TIMEOUT_S = 0.50
# 実測 Dex1 state (lab の rt/dex1、会場の gripper_q) の鮮度の上限 [s]。
# hand_pre_motion の state_max_age_s と同じ値。
HAND_STATE_STALE_TIMEOUT_S = 0.50
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


class _NeverEnterCheck(dict):
    """どの skill 名で引いても `enter_never` を返す enter_check の表。

    `[]` だけでなく `.get()` と `in` も同じにする (dict の `__missing__` は `[]` でしか
    呼ばれないので、`.get()` だと None が返り、呼んだ所で落ちる)。
    """

    def __missing__(
        self, skill_name: str
    ) -> Callable[[list[OBBDetection], SkillState], bool]:
        return enter_never

    def get(  # type: ignore[override]
        self, skill_name: str, default: object = None
    ) -> Callable[[list[OBBDetection], SkillState], bool]:
        return self[skill_name]

    def __contains__(self, skill_name: object) -> bool:
        return True


def stage_enter_check() -> dict[str, Callable[[list[OBBDetection], SkillState], bool]]:
    """stage の run (会場と自前実機の ``--stage``) の enter_check。YOLO では次へ進まない。

    上位 policy (YOLO の ``enter_*``) は、pick の途中など skill が終わる前に次の
    skill へ進めることがあったので外す (2026-09-24 ユーザー決定)。stage の run は
    skill の完了 (``is_complete``) と時間切れ (``timeout_reason`` /
    ``max_seconds_hard``) だけで進む (``advance_finished_skill``)。

    YOLO の検出そのものは止めない。``tick()`` は検出を policy 用の filter にも通し、
    ``obs["cleaned"]`` として skill に渡す (台を回す model と hybrid の VLM の
    overlay)。この表が決めるのは「次へ進むか」だけ。

    表に無い名前も ``enter_never`` になるので、stage に skill を足しても YOLO の
    判定が紛れ込まず、登録漏れで KeyError にもならない。
    """
    return _NeverEnterCheck()


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
    # Stage 0 waits for go-live first, lowers the arms and walks, then runs the rest
    # of the head procedure for the first pick (build_stage_skill_sequence).
    # Issue #141 (D7-1) moved the per-skill pre-motion into that head procedure, so
    # these lists hold only the skills that are specific to the stage itself.
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
LEARNED_STAGE_SKILLS = frozenset(
    {
        "rotate_table_base",
        "pick_table_leg",
        "insert_table_leg",
        "rotate_leg_to_tighten",
        "flip_table",
    }
)
STAGE_MIN: int = 0
STAGE_MAX: int = 5


def build_stage_transitions(
    stage: int,
    *,
    is_start_stage: bool = True,
    include_hand: bool = True,
    skip_model_transition_pairs: frozenset[tuple[str, str]] = frozenset(),
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
        stage,
        is_start_stage=is_start_stage,
        include_hand=include_hand,
        skip_model_transition_pairs=skip_model_transition_pairs,
    )
    transitions: dict[str, list[str]] = {}
    for i, skill in enumerate(skills):
        transitions[skill] = [skills[i + 1]] if i + 1 < len(skills) else []
    return transitions


def build_stage_skill_sequence(
    stage: int,
    *,
    is_start_stage: bool = True,
    include_hand: bool = True,
    skip_model_transition_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> list[str]:
    """Stage で dispatch する skill 名の列 (頭の手順を含む、Issue #141 D7-1)。

    頭の手順 (手を開く → 腕の pre-motion → 手を開始の開度へ → N 秒保持) は、
    次に始める学習 skill の開始姿勢へ体を持っていく手順。

    - stage 0: go-live 待ちだけを先頭に置き (腕を下ろす・歩くのはロボットが指令に
      従い始めてから。go-live 前の adapter は WBC に何も送らない)、残りの頭の手順
      (手を開く → 腕の pre-motion → …) は歩いた後に行う
    - stage 1: stage 0 で済んでいるので入れない。**stage 1 から起動したときだけ**先頭に入れる
    - stage 2〜5: 前の脚の締め終わりから始まるので、必ず先頭に入れる

    Args:
        stage: STAGE_SKILL_SEQUENCES の key (0..5)。
        is_start_stage: この stage から起動したか (途中から再開したか)。
        include_hand: Dex1 を実際に動かすか。False なら手の skill を列に入れない。
    """
    base = list(STAGE_SKILL_SEQUENCES[stage])
    expanded: list[str] = []
    for index, skill_name in enumerate(base):
        expanded.append(skill_name)
        if index + 1 >= len(base):
            continue
        next_skill = base[index + 1]
        if (
            skill_name not in LEARNED_STAGE_SKILLS
            or next_skill not in LEARNED_STAGE_SKILLS
        ):
            continue
        if (skill_name, next_skill) in skip_model_transition_pairs:
            expanded.extend(self_bridged_transition_skill_names(skill_name, next_skill))
            continue
        expanded.extend(
            model_transition_skill_names(
                skill_name, next_skill, include_hand=include_hand
            )
        )
    base = expanded
    head_skill = STAGE_HEAD_SKILL.get(stage)
    if stage == 1 and not is_start_stage:
        head_skill = None
    if stage >= 2 and not is_start_stage and head_skill is not None:
        previous_skill = STAGE_SKILL_SEQUENCES[stage - 1][-1]
        pair = (previous_skill, head_skill)
        if pair not in skip_model_transition_pairs:
            return (
                model_transition_skill_names(
                    previous_skill, head_skill, include_hand=include_hand
                )
                + base
            )
        return self_bridged_transition_skill_names(previous_skill, head_skill) + base
    if head_skill is None:
        return base
    head = head_procedure_skill_names(head_skill, include_hand=include_hand)
    if stage == 0:
        return head[:1] + base + head[1:]
    return head + base


# 脚を持ったまま次の model へ運ぶ境界 (Issue #159 T3)。ここだけは手をそのままにして
# 腕を先に動かす。それ以外の境界は既定で、腕を動かす前に握っている手を離す
# (`hand_release_*`)。境界が増えても、ここに足さなければ離す側に倒れる。
#
# 教師 episode は pick 以外、手を離して終わる (最後の frame で握っているのは
# insert / rotate_leg / rotate_table_base とも 0〜8%、2026-09-23 に dataset で確認)。
# 危ないのは離す前に時間切れで切られたとき (大会経路の rotate_leg は時間切れでしか
# 終わらない) で、そのまま腕を上げると挿した脚や締めた脚を引き上げる。
KEEP_GRIP_MODEL_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {("pick_table_leg", "insert_table_leg")}
)


def model_transition_skill_names(
    previous_skill: str, next_skill: str, *, include_hand: bool = True
) -> list[str]:
    """Finite transition steps inserted between two learned models.

    `KEEP_GRIP_MODEL_TRANSITIONS` (pick -> insert): the arm path runs first while
    the existing Dex1 publisher retains its last target, so the carried leg is
    not dropped.  Every other boundary first releases a hand that is still
    gripping (`hand_release_*`), then moves the arm.  In both cases the next
    model's frame-zero hand target is applied only after measured arm
    convergence.
    """

    suffix = f"{previous_skill}_to_{next_skill}"
    names = []
    if include_hand and (previous_skill, next_skill) not in KEEP_GRIP_MODEL_TRANSITIONS:
        names.append(f"hand_release_{suffix}")
    names.append(f"arm_transition_{suffix}")
    if include_hand:
        names.append(f"hand_transition_{suffix}")
    names.append(f"hold_transition_{suffix}")
    return names


def self_bridged_transition_skill_names(
    previous_skill: str, next_skill: str
) -> list[str]:
    """自分で次の model の frame-zero まで運ぶ境界 (hybrid pick -> insert) の手順。

    腕・手の移動は前の skill が済ませているので省くが、**HOLD だけは残す**
    (Codex #5)。model の載せ替え (前を解放 → 次を読む → 準備確認) は HOLD の中で
    しか起きない (`ModelResidency.on_skill_started`)。HOLD が無いと次の skill の
    開始 (`reset`) が制御 thread で同期に読み込み、前の model を載せたまま
    2 個が GPU に同時に載っていた。
    """
    return [f"hold_transition_{previous_skill}_to_{next_skill}"]


def model_transition_pairs_for_stage(
    stage: int,
    *,
    is_start_stage: bool,
    skip_model_transition_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[tuple[str, str], ...]:
    """Return model boundaries whose finite transition skills must be built."""

    if stage not in STAGE_SKILL_SEQUENCES:
        raise KeyError(
            f"stage {stage} not registered (valid: {sorted(STAGE_SKILL_SEQUENCES)})"
        )
    base = STAGE_SKILL_SEQUENCES[stage]
    pairs = [
        pair
        for pair in zip(base, base[1:])
        if pair[0] in LEARNED_STAGE_SKILLS and pair[1] in LEARNED_STAGE_SKILLS
    ]
    head_skill = STAGE_HEAD_SKILL.get(stage)
    if stage >= 2 and not is_start_stage and head_skill is not None:
        pairs.insert(0, (STAGE_SKILL_SEQUENCES[stage - 1][-1], head_skill))
    return tuple(pair for pair in pairs if pair not in skip_model_transition_pairs)


def validate_stage_model_transition_chain(
    stage: int,
    sequence: list[str],
    *,
    is_start_stage: bool,
    include_hand: bool = True,
    self_bridged_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> None:
    """Fail closed unless every learned-policy boundary has a pose scaffold.

    A normal boundary must be exactly ``previous -> arm transition -> optional
    Dex1 transition -> stable hold -> next``.  A pair may be omitted only when
    it is explicitly declared self-bridging (currently the hybrid pick skill,
    whose terminal FSM state is the insert model's measured frame-zero pose).

    Continuous Stage 2--5 execution starts the new stage with a transition from
    the previous stage's final model, so that boundary is validated as a prefix
    even though the previous model belongs to the preceding orchestrator run.
    """

    expected_pairs = model_transition_pairs_for_stage(
        stage,
        is_start_stage=is_start_stage,
        skip_model_transition_pairs=frozenset(),
    )
    for previous_skill, next_skill in expected_pairs:
        pair = (previous_skill, next_skill)
        if pair in self_bridged_pairs:
            try:
                previous_index = sequence.index(previous_skill)
            except ValueError:
                # Cross-stage self bridges are not currently used.  Reject one
                # rather than accepting an unverifiable implicit transition.
                raise ValueError(
                    f"self-bridged model boundary {pair!r} has no previous policy "
                    "in the stage sequence"
                ) from None
            expected = self_bridged_transition_skill_names(*pair) + [next_skill]
            actual = sequence[previous_index + 1 : previous_index + 1 + len(expected)]
            if actual != expected:
                raise ValueError(
                    f"self-bridged model boundary {pair!r} must be exactly "
                    f"{expected!r} (the model switch HOLD), got {actual!r}"
                )
            continue

        scaffold = model_transition_skill_names(
            previous_skill, next_skill, include_hand=include_hand
        )
        expected = scaffold + [next_skill]
        if sequence[: len(expected)] == expected:
            # Cross-stage boundary.  The same policy name may also occur at the
            # end of this stage (for example tighten -> next rotate ... ->
            # tighten), so prefix detection must take precedence over index().
            actual = sequence[: len(expected)]
        elif previous_skill in sequence:
            start = sequence.index(previous_skill) + 1
            actual = sequence[start : start + len(expected)]
        else:
            # A continuous stage boundary begins with the scaffold; the prior
            # policy completed in the preceding stage runner.
            actual = sequence[: len(expected)]
        if actual != expected:
            raise ValueError(
                "learned-policy boundary is missing its verified initial-pose "
                f"transition: pair={pair!r} expected={expected!r} actual={actual!r}"
            )

    # No undeclared direct learned-policy adjacency is allowed.  This catches a
    # newly added policy pair even if its author forgets to update the pair list.
    for previous_skill, next_skill in zip(sequence, sequence[1:]):
        pair = (previous_skill, next_skill)
        if (
            previous_skill in LEARNED_STAGE_SKILLS
            and next_skill in LEARNED_STAGE_SKILLS
            and pair not in self_bridged_pairs
        ):
            raise ValueError(
                f"direct learned-policy transition is forbidden: {pair!r}"
            )


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
        reason: "ended" (目標に届かないまま終わった) / "complete" / "timeout" /
            "dwell" のどれで進んだか。
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

        # 観測の鮮度は **skill を start する前に**確かめる (Codex #6)。start は
        # 歩行の速度指令など外部への指令を伴うことがあり、以前は古い画像で
        # 例外になる前に vx が出ていた。
        obs = self._build_obs(frame, self._policy_cleaned)
        self._check_observation_freshness(obs)
        started = False

        # a) 初回 tick で initial skill を dispatcher で auto start
        if self.dispatcher.active_skill_name is None:
            self.dispatcher.start(
                self.state.current_skill,
                self._build_params(self.state.current_skill),
            )
            started = True

        # c) 上位 Planner の判定は、cleaner の出力が出た tick だけ。溜まっていない間も
        #    skill は止めない (Issue #141 D3)。
        fired_to: Optional[str] = None
        if planner_cleaned is not None:
            # VlaSkill の progress monitor は policy と同じ current-frame 検出を使う。
            # 直前 tick の結果を Planner state へ同期し、次 tick の遷移判定に使う。
            # 1 tick (約 33ms) の遅延と引き換えに、画像 / overlay と判定の時刻を揃える。
            active_skill = self.dispatcher.active_skill
            last_action = getattr(active_skill, "last_action", None)
            metadata = getattr(last_action, "metadata", None)
            if isinstance(metadata, dict):
                self.state.update_table_rotation_progress(metadata)
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
                if prev_skill == "rotate_table_base":
                    self._log_control_event(
                        "rotate_success_transition",
                        skill=prev_skill,
                        next_skill=fired_to,
                        evidence=(
                            "validated_progress_monitor"
                            if self.state.table_rotation_monitor_available
                            else "legacy_replay_fallback"
                        ),
                        table_rotation_deg=self.state.table_rotation_deg,
                        table_rotation_usable=self.state.table_rotation_usable,
                        table_rotation_sustained=self.state.table_rotation_sustained,
                    )
                print(
                    f"[orch] fire transition: {prev_skill} -> {fired_to}",
                    file=sys.stderr,
                )

        # d) 観測 → skill (評価経路と同じ組み立て、Issue #141 D3)。start に時間が
        #    掛かった tick (model の読み込み等) は観測を取り直して再確認する。
        if started or fired_to is not None:
            obs = self._build_obs(frame, self._policy_cleaned)
            self._check_observation_freshness(obs)
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

    def _check_observation_freshness(self, obs: dict) -> None:
        """カメラ・関節 state・実測 Dex1 state。指令を出す処理の前に必ず通す。"""
        self._check_camera_freshness(obs)
        self._check_joint_state_freshness(obs)
        self._check_hand_state_freshness(obs)

    def _check_hand_state_freshness(self, obs: dict) -> None:
        """実測の Dex1 state が古ければ安全停止する (合成値は対象外)。

        公式 `:5557` の `gripper_q` は一度受けたら実測として扱う。途絶えても
        `get()` は最後の値を返すので、ここで止めないと policy の入力と把持判定が
        古い開度のまま進む。
        """
        state = obs.get("hand_state")
        if state is None or not bool(getattr(state, "measured", False)):
            return
        received = [
            getattr(state, "left_received_monotonic_ns", None),
            getattr(state, "right_received_monotonic_ns", None),
        ]
        if any(value is None for value in received):
            return
        age_s = (time.monotonic_ns() - min(int(v) for v in received)) * 1e-9
        if age_s > HAND_STATE_STALE_TIMEOUT_S:
            raise LiveSourceSafetyError(
                f"measured Dex1 state is stale ({age_s:.3f}s > "
                f"{HAND_STATE_STALE_TIMEOUT_S:g}s)"
            )

    def _check_joint_state_freshness(self, obs: dict) -> None:
        """受信時刻を持つ関節 state が古ければ安全停止に回す。"""
        state = obs.get("joint_state")
        received_ns = getattr(state, "received_monotonic_ns", None)
        if received_ns is None:
            return
        age_s = (time.monotonic_ns() - int(received_ns)) * 1e-9
        if age_s > JOINT_STATE_STALE_TIMEOUT_S:
            raise LiveSourceSafetyError(
                f"joint state is stale ({age_s:.3f}s > "
                f"{JOINT_STATE_STALE_TIMEOUT_S:g}s)"
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

            # failure_reason / is_complete / max_seconds_hard / max_dwell_sec。boundary の driver も
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
                # 遅れた周は「今」から数え直す (次の周の += dt で今 + dt)。ここで dt を
                # 足すと次の周が dt を 2 つ待ち、40 ms の処理が 40/66.7 ms 交互 = 19.2 Hz
                # に落ちる (B2a-06)。遅れを取り戻す短い間隔は、どちらでも出ない。
                next_deadline = time.monotonic()

    def advance_finished_skill(self, now: Optional[float] = None) -> SkillAdvance:
        """active skill が終わったか時間切れなら次へ進める。1 回の呼び出しで最大 1 遷移。

        `tick()` は **YOLO の `enter_check` しか見ない**。この method はその受け皿で、
        5 つの signal を順に見る:

          0. `failure_reason`   故障 (model・センサー・Dex1 が動かない、入力が読めない、
                                設定ミス)。**必ず HOLD し、次へ進まない**。
          1. `is_complete`      skill が自分で「収束した」と言う (腕の pre-motion 等)
          1b. `timeout_reason`  有限手順 (腕・手の準備動作、hybrid の持ち替え) が自分の
                                締め切りまでに目標へ届かなかった。故障ではないので
                                **時間切れと同じく次へ進む** (止めるのは人)
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
            LiveSourceSafetyError: 有限手順が故障で止まった、`on_timeout: stop` の skill が
                時間切れした、または dwell 上限に達したのに遷移先が無い。
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

        def _advance(
            cand: str, reason: str, message: str, detail: Optional[str] = None
        ) -> SkillAdvance:
            if reason == "timeout":
                # 時間切れで進んだ遷移を、成功で進んだ遷移と機械的に分けられるよう
                # JSONL に残す。端末には 1 行出るだけで、run の後で読み返せない。
                self._log_control_event(
                    "skill_timeout_fallback",
                    skill=current_active,
                    timeout_s=self.hard_timeout_by_skill.get(current_active or ""),
                    next_skill=cand,
                    **({"detail": detail} if detail else {}),
                )
            ctx = self._build_transition_ctx(cand)
            self.state.transition(cand, ctx)
            self.dispatcher.start(cand, self._build_params(cand))
            print(message, file=sys.stderr)
            self._active_skill_dwell_fired = True
            return SkillAdvance(fired_to=cand, reason=reason)

        # 0) ``failure_reason`` は故障 (model・センサー・Dex1 が動かない、operator
        #    gate の入力が読めない、設定ミス)。次へ進めても動かないので HOLD して止める。
        #    目標に届かないまま締め切りを過ぎただけのものは故障ではない。skill は
        #    ``timeout_reason`` で返し、下の 1b で次へ進む (stage の run で次へ進む道は
        #    完了と時間切れだけ、止めるのは人)。
        failure_reason = (
            getattr(active_skill_obj, "failure_reason", None)
            if active_skill_obj is not None
            else None
        )
        if failure_reason and not self._active_skill_dwell_fired:
            raise LiveSourceSafetyError(
                f"skill {current_active!r} failed before reaching its verified "
                f"target; holding and refusing the next skill: {failure_reason}"
            )

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

        # 1b) 有限手順が自分の締め切りまでに目標へ届かなかった。YAML の時間切れと
        #     同じ扱いで次へ進む。理由は JSONL (detail) と stderr に残す。
        timeout_reason = (
            getattr(active_skill_obj, "timeout_reason", None)
            if active_skill_obj is not None
            else None
        )
        if timeout_reason and not self._active_skill_dwell_fired:
            candidates = self.transitions.get(current_active, [])
            if not candidates:
                self._log_control_event(
                    "skill_timeout_fallback",
                    skill=current_active,
                    timeout_s=None,
                    next_skill=None,
                    detail=str(timeout_reason),
                )
                print(
                    f"[orch] terminal skill {current_active!r} timed out short of "
                    f"its target ({timeout_reason}); stage finished",
                    file=sys.stderr,
                )
                return SkillAdvance(reason="timeout", stage_finished=True)
            return _advance(
                candidates[0],
                "timeout",
                f"[orch] skill {current_active} timed out short of its target "
                f"({timeout_reason}); moving on: {current_active} -> {candidates[0]}",
                detail=str(timeout_reason),
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
            "table_rotation_monitor_available": (
                self.state.table_rotation_monitor_available
            ),
            "table_rotation_usable": self.state.table_rotation_usable,
            "table_rotation_sustained": self.state.table_rotation_sustained,
            "table_rotation_deg": self.state.table_rotation_deg,
        }
        self.log_sink.write(json.dumps(payload) + "\n")  # type: ignore[union-attr]

    def _log_control_event(self, event: str, **fields: object) -> None:
        """tick の記録とは別に、遷移の理由を同じ JSONL に残す (Issue #152)。"""
        if self.log_sink is None:
            return
        payload = {
            "event": event,
            "t_monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        self.log_sink.write(json.dumps(payload) + "\n")
        self.log_sink.flush()

    # Resume mechanism (restore_from_snapshot / _emit_transition_snapshot) は
    # Issue #128 Phase 3 完了に伴い削除。--stage=N で再実行が中断復旧の代替。
