"""State machine の enter 関数群。

6-skill list:
    move_to_table (tid=1) → move_table_base (tid=5 or 7 merged) → pick_table_leg (tid=4)
    → insert_table_leg (tid=0) → rotate_leg_to_tighten (tid=3)
    → (loop back to move_table_base) or flip_table (tid=2)

Design:
    - pick_table_leg: 1st も 2nd 以降も同 signature、unified single function 内で state
      によって Kabsch / aspect を切替
    - move_table_base: 1st (初期 approach 後) と 2nd 以降 (leg secured 後) で条件違うので
      parent + child (first/next) 構造

閾値は `configs/thresholds.yaml` で外出し (code change なしに iter tune 可能)。
"""

from __future__ import annotations

from pathlib import Path

import yaml

import numpy as np

from inference.desktop.perception.yolo_obb import OBBDetection
from inference.desktop.skill_planner.geometry import (
    aabb_reaches_obb_left_edge,
    bottom_left_vertex_y,
    kabsch_rotation_angle_deg,
    obb_aabb_iou,
    obb_aspect_ratio,
)
from inference.desktop.skill_planner.state import SkillState

# ---- thresholds を YAML から load (module import 時、1 回のみ) ----
_THRESHOLDS_PATH = Path(__file__).parent / "configs" / "thresholds.yaml"


def _load_thresholds() -> dict:
    if not _THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"thresholds config not found: {_THRESHOLDS_PATH}")
    return yaml.safe_load(_THRESHOLDS_PATH.read_text())


_TH = _load_thresholds()
PICK_FIRST_LEG_TABLE_TOP_ANGLE_DELTA_MIN: float = float(
    _TH["pick_first_leg_table_top_angle_delta_min"]
)
PICK_NEXT_LEG_TABLE_TOP_ASPECT_RATIO_MIN: float = float(
    _TH["pick_next_leg_table_top_aspect_ratio_min"]
)
PICK_LEG_REF_N_AVG_FRAMES: int = int(_TH["pick_leg_ref_n_avg_frames"])
MOVE_TABLE_BASE_NEXT_HAND_LEFT_TABLE_TOP_LEFT_MAX_DIST: float = float(
    _TH["move_table_base_next_hand_left_table_top_left_max_dist"]
)
MOVE_TABLE_BASE_NEXT_HAND_IN_TABLE_FRAC_MAX: float = float(
    _TH["move_table_base_next_hand_in_table_frac_max"]
)
FLIP_HAND_LEFT_TABLE_TOP_LEFT_MAX_DIST: float = float(
    _TH["flip_hand_left_table_top_left_max_dist"]
)
FLIP_HAND_IN_TABLE_FRAC_MAX: float = float(
    _TH["flip_hand_in_table_frac_max"]
)
INSERT_RIGHT_HAND_LEG_IOU_MIN: float = float(_TH["insert_right_hand_leg_iou_min"])
INSERT_LEFT_HAND_LEG_IOU_MAX: float = float(_TH["insert_left_hand_leg_iou_max"])
LEFT_HAND_TOUCHED_LEG_IOU_SEEN_MIN: float = float(
    _TH["left_hand_touched_leg_iou_seen_min"]
)
LEG_TIP_HOLE_IOU_SEEN_MIN: float = float(_TH["leg_tip_hole_iou_seen_min"])
LEFT_RIGHT_HAND_IOU_SEEN_MIN: float = float(_TH["left_right_hand_iou_seen_min"])
ROTATE_LEG_RIGHT_HAND_BL_VS_TABLE_TOP_BOTTOM_MAX_DELTA: float = float(
    _TH["rotate_leg_right_hand_bl_vs_table_top_bottom_max_delta"]
)


def _table_top_verts(dets: list[OBBDetection]) -> np.ndarray | None:
    """confidence 最大の table_top OBB の verts を返す。

    Args:
        dets: 現 frame の OBB detection list。

    Returns:
        shape (4, 2) の verts、table_top 無検出は None。
    """
    tops = [d for d in dets if d.class_name == "table_top"]
    if not tops:
        return None
    top = max(tops, key=lambda d: d.confidence)
    return top.verts


def _max_iou(dets: list[OBBDetection], class_a: str, class_b: str) -> float:
    """class_a と class_b の全 detection pair の AABB IoU 最大値を返す。

    Args:
        dets: 現 frame の OBB detection list。
        class_a: 片方の class 名。
        class_b: もう片方の class 名。

    Returns:
        max IoU、どちらかの class が無検出なら 0.0。
    """
    boxes_a = [d for d in dets if d.class_name == class_a]
    boxes_b = [d for d in dets if d.class_name == class_b]
    if not boxes_a or not boxes_b:
        return 0.0
    best = 0.0
    for a in boxes_a:
        for b in boxes_b:
            iou = obb_aabb_iou(a.verts, b.verts)
            if iou > best:
                best = iou
    return best


# =========================================================================
# pick_table_leg
# =========================================================================
def enter_pick_table_leg(
    dets: list[OBBDetection],
    state: SkillState,
) -> bool:
    """table_top の姿勢変化から pick_table_leg 開始タイミングを検出する。

    state.n_legs_completed で signal を切替 (hybrid rule):
        1st pick (n=0): base_run が直線移動 (tid=7) で回転量小さい (~20°) →
            |Kabsch(ref, cur)| >= PICK_FIRST_LEG_TABLE_TOP_ANGLE_DELTA_MIN で fire。
        2nd+ pick (n=1,2,3): base_run が大回転 (tid=5) で aspect 変化明確 →
            cur_aspect / ref_aspect >= PICK_NEXT_LEG_TABLE_TOP_ASPECT_RATIO_MIN で fire。

    全 pick で Kabsch にすると ep507 2nd pick が peak 58° で threshold 60° 未達
    (0 fire)、aspect のみだと 1st pick は回転量小さすぎて発火不足で hybrid が必要。

    ref = base_run 開始直前 N frame の table_top verts の pivot-aligned mean
    (`state.base_rotation_start_table_top_verts`、orchestrator が preprocess)。

    Guard: 4 leg 完了後は fire しない。「pick 中の再 pick 抑止」は Orchestrator の
    TRANSITIONS graph (move_table_base → pick_table_leg のみ) が enforce する。

    Args:
        dets: 現 frame の OBB detection list。
        state: skill state machine の shared state。

    Returns:
        pick_table_leg に遷移すべき frame なら True。
    """
    if state.n_legs_completed >= 4:
        return False
    if state.base_rotation_start_table_top_verts is None:
        return False
    cur = _table_top_verts(dets)
    if cur is None:
        return False
    if state.n_legs_completed == 0:
        # 1st: Kabsch angle
        angle = kabsch_rotation_angle_deg(
            state.base_rotation_start_table_top_verts, cur
        )
        return abs(angle) >= PICK_FIRST_LEG_TABLE_TOP_ANGLE_DELTA_MIN
    else:
        # 2nd+: aspect ratio
        ref_aspect = obb_aspect_ratio(state.base_rotation_start_table_top_verts)
        cur_aspect = obb_aspect_ratio(cur)
        return cur_aspect / ref_aspect >= PICK_NEXT_LEG_TABLE_TOP_ASPECT_RATIO_MIN


# =========================================================================
# move_table_base (n_legs_completed で dispatch、1st cycle は dwell timer で入るため skip)
# =========================================================================
def enter_move_table_base(dets: list[OBBDetection], state: SkillState) -> bool:
    """n_legs_completed に応じて enter 判定する。

    Dispatch:
        - n_legs_completed == 0 → False (1st cycle は move_to_table の dwell timer
          で入るため YOLO 判定 skip、orchestrator.py auto-transition 参照)。
          将来 rotate_table_base skill 分離時に YOLO signal を復活させる余地として
          分岐は明示的に残す。
        - n_legs_completed 1..3 → _enter_move_table_base_next (loop iteration)

    Guard: 4 leg 完了後は fire しない。「pick 中の再 move_base 抑止」は Orchestrator の
    TRANSITIONS graph (rotate_leg_to_tighten → move_table_base のみ) が enforce する。

    Args:
        dets: 現 frame の OBB detection list。
        state: skill state machine の shared state。

    Returns:
        move_table_base に遷移すべき frame なら True。
    """
    if state.n_legs_completed >= 4:
        return False
    if state.n_legs_completed == 0:
        return False
    return _enter_move_table_base_next(dets)


def _enter_move_table_base_next(dets: list[OBBDetection]) -> bool:
    """前 leg 完了後 (n_legs=1..3) の loop iteration 発火判定。

    hand_left AABB が table_top OBB の左辺に「到達」で fire。到達条件 (OR):
        (a) 左辺 segment と hand AABB の最短距離 <= threshold
        (b) 0 < (hand ∩ table 面積 / hand 面積) <= threshold
            (hand が table に部分重なり = 端を掴んでる姿勢、full-inside は除外)

    (b) は rot_leg 中は hand が leg を掴んで table 深く入る (frac ≈ 0.9-1.0)、
    base 回転支え準備で edge に移動すると frac が閾値以下に落ちる、という物理観察。
    (a) 単独では hand が edge を完全に跨いだ場合を逃すので (b) と OR で確実発火化。

    物理的意味: 左手が table_top の左端に達した瞬間 = base 回転支え準備。

    Args:
        dets: 現 frame の OBB detection list。

    Returns:
        hand_left が table_top 左辺に到達していれば True。
        table_top / hand_left いずれか未検出は False。
    """
    tops = [d for d in dets if d.class_name == "table_top"]
    lefts = [d for d in dets if d.class_name == "hand_left"]
    if not tops or not lefts:
        return False
    t = max(tops, key=lambda d: d.confidence).verts
    l = max(lefts, key=lambda d: d.confidence).verts
    return aabb_reaches_obb_left_edge(
        t,
        float(l[:, 0].min()), float(l[:, 1].min()),
        float(l[:, 0].max()), float(l[:, 1].max()),
        MOVE_TABLE_BASE_NEXT_HAND_LEFT_TABLE_TOP_LEFT_MAX_DIST,
        MOVE_TABLE_BASE_NEXT_HAND_IN_TABLE_FRAC_MAX,
    )


# =========================================================================
# rotate_leg_to_tighten
# =========================================================================
def enter_rotate_leg_to_tighten(
    dets: list[OBBDetection], state: SkillState
) -> bool:
    """3 条件 (履歴 2 + 現 frame 1) で rotate_leg_to_tighten 開始を検出する。

        (1) state.leg_tip_touched_hole_since_pick (leg 挿入完了 proxy、履歴 flag)
        AND
        (2) state.left_right_hand_overlapped_since_pick (dual-hand grip 再結合、履歴 flag)
        AND
        (3) |hand_right の左下 vertex y - table_top 最下 y|
            < ROTATE_LEG_RIGHT_HAND_BL_VS_TABLE_TOP_BOTTOM_MAX_DELTA
            (右手 OBB の左下頂点が table_top 下辺の高さに到達した瞬間)

    Guard: 4 leg 完了後は fire しない。「leg 保持中のみ fire」は Orchestrator の
    TRANSITIONS graph (insert_table_leg → rotate_leg_to_tighten のみ) が enforce する。

    Args:
        dets: 現 frame の OBB detection list。
        state: skill state machine の shared state。

    Returns:
        rotate_leg_to_tighten に遷移すべき frame なら True。
    """
    if state.n_legs_completed >= 4:
        return False
    if not state.leg_tip_touched_hole_since_pick:
        return False
    if not state.left_right_hand_overlapped_since_pick:
        return False

    rights = [d for d in dets if d.class_name == "hand_right"]
    tops = [d for d in dets if d.class_name == "table_top"]
    if not rights or not tops:
        return False
    r = max(rights, key=lambda d: d.confidence)
    t = max(tops, key=lambda d: d.confidence)
    right_bl_y = bottom_left_vertex_y(r.verts)
    table_bottom = float(t.verts[:, 1].max())
    return abs(right_bl_y - table_bottom) < ROTATE_LEG_RIGHT_HAND_BL_VS_TABLE_TOP_BOTTOM_MAX_DELTA


# =========================================================================
# insert_table_leg
# =========================================================================
def enter_insert_table_leg(
    dets: list[OBBDetection], state: SkillState
) -> bool:
    """dual-hand pick → 左手 release → insert 準備完了 の temporal 判定。

        右手 ∩ leg IoU > INSERT_RIGHT_HAND_LEG_IOU_MIN (現 frame で掴んでる)
        AND state.left_hand_touched_leg_since_pick (過去に左手も触れてた)
        AND 左手 ∩ leg IoU < INSERT_LEFT_HAND_LEG_IOU_MAX (現 frame では離れてる)

    Guard: 4 leg 完了後は fire しない。「leg 保持中のみ fire」は Orchestrator の
    TRANSITIONS graph (pick_table_leg → insert_table_leg のみ) が enforce する。

    Args:
        dets: 現 frame の OBB detection list。
        state: skill state machine の shared state。

    Returns:
        insert_table_leg に遷移すべき frame なら True。
    """
    if state.n_legs_completed >= 4:
        return False
    if not state.left_hand_touched_leg_since_pick:
        return False
    right_iou = _max_iou(dets, "hand_right", "leg")
    left_iou = _max_iou(dets, "hand_left", "leg")
    return (
        right_iou > INSERT_RIGHT_HAND_LEG_IOU_MIN
        and left_iou < INSERT_LEFT_HAND_LEG_IOU_MAX
    )


# =========================================================================
# flip_table (move_table_base_next と signal 共通、state guard のみ差)
# =========================================================================
def enter_flip_table(dets: list[OBBDetection], state: SkillState) -> bool:
    """4th rotate_leg 中に flip_table への遷移タイミングを検出する。

    hand_left AABB が table_top OBB の左辺に「到達」で fire。到達条件 (OR):
        (a) 左辺 segment と hand AABB の最短距離 <= threshold
        (b) 0 < (hand ∩ table 面積 / hand 面積) <= threshold

    Signal は _enter_move_table_base_next と完全同一、state guard のみ差
    (move_base_next=n_legs 1..3、flip=n_legs==3)。右手ベースは 4th rot_leg 中も
    leg 掴んで discriminator にならないので左手 signal を採用。

    Guard: n_legs_completed == 3 (3 leg 完了、4th 進行中) でのみ fire。
    enter_flip 発火 = 4th rotate_leg を止めて flip に遷移するトリガーなので、
    「n_legs=4 になった後」ではなく「n_legs=3 の 4th cycle 中」に発火させる必要がある。
    n_legs>=4 guard にすると circular (4th rot_leg 完了に enter_flip 発火が必要かつ
    enter_flip 発火に n_legs=4 が必要) で永遠に fire できない。

    Args:
        dets: 現 frame の OBB detection list。
        state: skill state machine の shared state。

    Returns:
        flip_table に遷移すべき frame なら True。
    """
    if state.n_legs_completed != 3:
        return False
    tops = [d for d in dets if d.class_name == "table_top"]
    lefts = [d for d in dets if d.class_name == "hand_left"]
    if not tops or not lefts:
        return False
    t = max(tops, key=lambda d: d.confidence).verts
    l = max(lefts, key=lambda d: d.confidence).verts
    return aabb_reaches_obb_left_edge(
        t,
        float(l[:, 0].min()), float(l[:, 1].min()),
        float(l[:, 0].max()), float(l[:, 1].max()),
        FLIP_HAND_LEFT_TABLE_TOP_LEFT_MAX_DIST,
        FLIP_HAND_IN_TABLE_FRAC_MAX,
    )
