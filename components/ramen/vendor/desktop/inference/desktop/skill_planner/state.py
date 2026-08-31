"""State machine の shared state (skill 遷移判定に使う frame-persistent 情報)。

`SkillState.update(cleaned)` は 1 frame ごとに履歴 flag を latch。
`SkillState.transition(new_skill, ctx)` は skill fire で atomically 副作用を実行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from inference.desktop.perception.yolo_obb import OBBDetection
from inference.desktop.skill_planner.geometry import obb_aabb_iou

# ---- latch threshold を YAML から load (module import 時、1 回のみ) ----
_THRESHOLDS_PATH = Path(__file__).parent / "configs" / "thresholds.yaml"


def _load_latch_thresholds() -> tuple[float, float, float]:
    th = yaml.safe_load(_THRESHOLDS_PATH.read_text())
    return (
        float(th["left_hand_touched_leg_iou_seen_min"]),
        float(th["leg_tip_hole_iou_seen_min"]),
        float(th["left_right_hand_iou_seen_min"]),
    )


_LEFT_HAND_LEG_IOU_MIN, _LEG_TIP_HOLE_IOU_MIN, _LEFT_RIGHT_HAND_IOU_MIN = (
    _load_latch_thresholds()
)

# 履歴 flag latch を許可する skill 集合。pick cycle 開始以降 次の pick 開始まで
# latch を受け付ける。move_to_table / move_table_base / flip_table 中は latch しない。
_CYCLE_PHASES: frozenset[str] = frozenset(
    {"pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"}
)


def _max_iou(
    dets: list[OBBDetection], class_a: str, class_b: str
) -> float:
    """class_a と class_b の全 detection pair の AABB IoU 最大値。無検出は 0.0。"""
    boxes_a = [d for d in dets if d.class_name == class_a]
    boxes_b = [d for d in dets if d.class_name == class_b]
    if not boxes_a or not boxes_b:
        return 0.0
    return max(
        obb_aabb_iou(a.verts, b.verts) for a in boxes_a for b in boxes_b
    )


@dataclass
class SkillState:
    """6-skill state machine の shared state。

    Attributes:
        current_skill: 現在 active な skill 名 (初期値 = "move_to_table")。
            Orchestrator の TRANSITIONS graph lookup key + update() の
            _CYCLE_PHASES latch guard に使う。
        n_legs_completed: 完了 leg 数 (0..4)。transition() 内で prev ==
            "rotate_leg_to_tighten" のとき +1 (move_table_base / flip_table
            どちらへの遷移でも rotate 完了扱い)。
            enter_pick_table_leg (n=0 → Kabsch, n>=1 → aspect) /
            enter_move_table_base (n=0 → first, n=1..3 → next) /
            enter_flip_table (n==3 でのみ fire) の分岐に使う。
        base_rotation_start_table_top_verts: 直近 move_table_base 遷移開始時の
            table_top OBB 4 頂点 (shape (4, 2), normalized [0, 1])。
            transition("move_table_base", ctx) 時に Orchestrator から
            N-frame ring の pivot-aligned mean を ctx で受け取って snapshot。
            enter_pick_table_leg の Kabsch / aspect 判定の reference。
        left_hand_touched_leg_since_pick: 現 pick cycle 中に左手 ∩ leg overlap
            が発生した履歴 flag。update() で latch (次 pick 開始で reset)。
            enter_insert_table_leg の「dual-hand pick → 左手 release」判定に
            使う (過去に触ってた記憶が要る)。
        leg_tip_touched_hole_since_pick: 現 pick cycle 中に leg_tip ∩ hole overlap
            が発生した履歴 flag。leg 挿入完了の proxy (深度センサー無しの補完)。
            enter_rotate_leg_to_tighten の必要条件。
        left_right_hand_overlapped_since_pick: 現 pick cycle 中に左手 ∩ 右手 overlap
            が発生した履歴 flag。dual-hand grip 再結合の記憶。
            enter_rotate_leg_to_tighten の必要条件。
    """

    current_skill: str
    n_legs_completed: int
    base_rotation_start_table_top_verts: Optional[np.ndarray] = None
    left_hand_touched_leg_since_pick: bool = False
    leg_tip_touched_hole_since_pick: bool = False
    left_right_hand_overlapped_since_pick: bool = False

    def update(self, cleaned: list[OBBDetection]) -> None:
        """1 frame の cleaned dets を受けて履歴 flag を latch。

        _CYCLE_PHASES (pick / insert / rotate) 中のみ latch を許可。
        一度 True になった flag は次 transition("pick_table_leg") で reset される
        まで持続 (latch 型)。
        """
        if self.current_skill not in _CYCLE_PHASES:
            return
        if _max_iou(cleaned, "hand_left", "leg") > _LEFT_HAND_LEG_IOU_MIN:
            self.left_hand_touched_leg_since_pick = True
        if _max_iou(cleaned, "leg_tip", "hole") > _LEG_TIP_HOLE_IOU_MIN:
            self.leg_tip_touched_hole_since_pick = True
        if (
            _max_iou(cleaned, "hand_left", "hand_right")
            > _LEFT_RIGHT_HAND_IOU_MIN
        ):
            self.left_right_hand_overlapped_since_pick = True

    def transition(self, new_skill: str, ctx: Optional[dict] = None) -> None:
        """skill fire で atomically 遷移。副作用:
            - current_skill 更新
            - new_skill == "pick_table_leg": 3 履歴 flag を全 reset (新 pick cycle 開始)
            - prev == "rotate_leg_to_tighten": n_legs_completed += 1
              (move_table_base / flip_table どちらへの遷移でも rotate 完了扱い)
            - ctx に "base_rotation_start_table_top_verts" があれば snapshot
              (Orchestrator が N-frame ring の pivot-aligned mean を渡す)
        """
        prev = self.current_skill
        self.current_skill = new_skill
        if new_skill == "pick_table_leg":
            self.left_hand_touched_leg_since_pick = False
            self.leg_tip_touched_hole_since_pick = False
            self.left_right_hand_overlapped_since_pick = False
        if prev == "rotate_leg_to_tighten":
            self.n_legs_completed += 1
        if ctx is not None and "base_rotation_start_table_top_verts" in ctx:
            self.base_rotation_start_table_top_verts = ctx[
                "base_rotation_start_table_top_verts"
            ]
