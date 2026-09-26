"""RAMEN-Ori skill_id 定義 + source `task_index` map (Issue #120 Phase A-3)。

# 対応表

`model/ramen_ori` の設計は num_skills=6 (design doc `ramen_ori_vla_design.md` §7)。
source dataset (BitRobot G1_WBT / curated chunks) の task_index (0..7、tid=1 skip、
tid=6 intro skip) を **curation policy に忠実に** skill_id 0..5 へマップする。

| source task_index | skill_name (curation)       | RAMEN-Ori skill_id |
|------------------:|-----------------------------|-------------------:|
|                 0 | insert_table_leg            |                  0 |
|                 2 | flip_table                  |                  1 |
|                 3 | rotate_leg_to_tighten       |                  2 |
|                 4 | pick_table_leg              |                  3 |
|                 5 | rotate_table_base           |                  4 |
|                 7 | move_table_base             |                  5 |

## 除外 (map しない)

- **tid=1 (move_to_table)**: SDK walk で代替、RAMEN-Ori 学習対象外
  (`docs/dataset/ramen_ori_curation_policy.md` 「skip 系」)
- **tid=6 (building_children_table)**: LeRobot 側 intro 用包括ラベル、動作系ではない

## 注記: inference 側との違い

`inference/desktop/skill_planner/enter_conditions.py` は tid=5 と tid=7 を **merge**
して "move_table_base" 1 skill として扱う (現行 stack)。RAMEN-Ori は curation
label に忠実に separate、学習側で 2 skill として区別 (design doc §3.4)。

## Phase 2 (Issue #120) の scope

- 実際に学習に使うのは **tid=5 (skill_id=4) と tid=7 (skill_id=5) の 2 skill のみ**
- 他 4 skill (0/2/3/4) は将来拡張用に skill_id 予約、Phase 2 では data 無し
- num_skills=6 のまま (embedding 未使用 4 個は死に weight)
"""

from __future__ import annotations


# skill_id 割当 (canonical)
SKILL_INSERT_TABLE_LEG = 0
SKILL_FLIP_TABLE = 1
SKILL_ROTATE_LEG_TO_TIGHTEN = 2
SKILL_PICK_TABLE_LEG = 3
SKILL_ROTATE_TABLE_BASE = 4
SKILL_MOVE_TABLE_BASE = 5

NUM_SKILLS = 6

# source task_index → RAMEN-Ori skill_id
TASK_INDEX_TO_SKILL_ID: dict[int, int] = {
    0: SKILL_INSERT_TABLE_LEG,
    2: SKILL_FLIP_TABLE,
    3: SKILL_ROTATE_LEG_TO_TIGHTEN,
    4: SKILL_PICK_TABLE_LEG,
    5: SKILL_ROTATE_TABLE_BASE,
    7: SKILL_MOVE_TABLE_BASE,
}

# skill_id → 表示用 name (log / debug 用)
SKILL_ID_TO_NAME: dict[int, str] = {
    SKILL_INSERT_TABLE_LEG: "insert_table_leg",
    SKILL_FLIP_TABLE: "flip_table",
    SKILL_ROTATE_LEG_TO_TIGHTEN: "rotate_leg_to_tighten",
    SKILL_PICK_TABLE_LEG: "pick_table_leg",
    SKILL_ROTATE_TABLE_BASE: "rotate_table_base",
    SKILL_MOVE_TABLE_BASE: "move_table_base",
}

# Phase 2 (Issue #120) 学習対象
PHASE2_SKILL_IDS: frozenset[int] = frozenset(
    {SKILL_ROTATE_TABLE_BASE, SKILL_MOVE_TABLE_BASE}
)


def task_index_to_skill_id(task_index: int) -> int:
    """source task_index → RAMEN-Ori skill_id。未定義 (tid=1/6/unknown) は raise。

    Args:
        task_index: source dataset の task_index (0-based int)

    Returns:
        RAMEN-Ori skill_id (0..NUM_SKILLS-1)

    Raises:
        KeyError: TASK_INDEX_TO_SKILL_ID に未定義 (tid=1, 6, or unexpected value)
    """
    if task_index not in TASK_INDEX_TO_SKILL_ID:
        raise KeyError(
            f"task_index={task_index} is not mapped to any RAMEN-Ori skill_id. "
            f"Known: {sorted(TASK_INDEX_TO_SKILL_ID)}. "
            f"tid=1 (move_to_table) and tid=6 (building_children_table) are intentionally skipped."
        )
    return TASK_INDEX_TO_SKILL_ID[task_index]


def skill_id_name(skill_id: int) -> str:
    """skill_id → 表示用 name (log/debug 用)。"""
    return SKILL_ID_TO_NAME.get(skill_id, f"unknown_skill_{skill_id}")
