"""boundary の driver が自前経路と同じ組み立てになっていることを確認する。

GPU / YOLO / model 不要 (policy は deferred のまま触らない)。

## なぜ必要か

driver は以前 `VlaCls(...)` を直接呼んでおり、`assembly.build_vla_skill` が
入れる 6 項目が丸ごと抜けていた (2026-09-21 に自前経路と突き合わせて判明):

    language_override / dispatch_waist / motion_limiter / teacher_range /
    skill 別 wrist FK / progress_monitor

とくに `rotate_table_base` は
  - variant の prompt が specialist 用の 'rotate table base' なのに
    class の "rotate and move table base (combined 5+7)" で推論していた
  - config が `dispatch_waist: False` なのに腰を出していた
  - 独自の `wrist_tool_offset` (既定と左手で 4.4cm 差) が効いていなかった

assembly を通せば、今後 skill_config に設定が増えても自動で入る。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_VENDOR_DESKTOP = _HERE.parent / "vendor" / "desktop"
for _p in (str(_VENDOR_DESKTOP), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components.ramen.orchestrator_driver import (  # noqa: E402
    _LEGS,
    _STAGE_SKILLS,
    _TRANSITIONS,
    _load_skill_config,
    _stage_variants,
)
from components.ramen.orchestrator_io import InterceptorActuator  # noqa: E402
from inference.desktop import assembly as _assembly  # noqa: E402
from inference.desktop.lower_policy.policies.config_loader import (  # noqa: E402
    load_policy_variant,
)
from inference.desktop.lower_policy.skills import vla_skill as _vla  # noqa: E402

_POLICY_CFG = str(
    _VENDOR_DESKTOP / "inference/desktop/lower_policy/configs/policy_config.yaml"
)


def _build(skill_name: str, cls_name: str, variant: str | None = None):
    """driver と同じ手順で 1 skill を組む。

    `variant` を省略すると driver と同じく `default_variant_by_skill` を読む
    (Issue #148: どの ckpt で走るかの正本は config)。
    """
    if variant is None:
        variant = _stage_variants(_POLICY_CFG)[skill_name]
    return _assembly.build_vla_skill(
        skill_name=skill_name,
        vla_skill_cls=getattr(_vla, cls_name),
        variant=load_policy_variant(_POLICY_CFG, variant),
        skill_config=_load_skill_config(str(_VENDOR_DESKTOP)),
        waist_actuator=InterceptorActuator("waist"),
        hand_actuator=InterceptorActuator("hand"),
        fk_factory=_assembly.FkFactory(),
        deferred=True,
    )


def test_every_stage_skill_gets_a_motion_limiter_and_teacher_range():
    """速度・位置の限界と教師の範囲の補正が全 skill に入ること。"""
    for skill_name, cls_name in _STAGE_SKILLS:
        skill = _build(skill_name, cls_name).skill
        assert getattr(skill, "_motion_limiter", None) is not None, skill_name
        assert getattr(skill, "_teacher_range", None) is not None, skill_name
        assert getattr(skill, "_fk", None) is not None, skill_name


def test_dispatch_waist_comes_from_the_skill_config():
    """`rotate_table_base` は False。True 固定だと腰を出してはいけない skill で出す。"""
    by_name = {
        name: _build(name, cls).dispatch_waist
        for name, cls in _STAGE_SKILLS
    }

    assert by_name["rotate_table_base"] is False
    assert by_name["pick_table_leg"] is True
    assert by_name["insert_table_leg"] is True
    assert by_name["rotate_leg_to_tighten"] is True


def test_rotate_table_base_uses_the_specialist_prompt():
    """variant の prompt が class の既定を上書きすること。"""
    entry = load_policy_variant(_POLICY_CFG, "groot_overlay")
    assert entry.policy_config.language_prompt == "rotate table base"

    skill = _build("rotate_table_base", "RotateTableBaseVlaSkill", "groot_overlay").skill

    # VlaSkill は `self._language_override or self.LANGUAGE` で決める (vla_skill.py:696)。
    assert skill._language_override == "rotate table base"
    assert skill._language_override != _vla.RotateTableBaseVlaSkill.LANGUAGE


def test_rotate_table_base_uses_its_own_wrist_offset():
    """skill 別の wrist_tool_offset が効くこと (既定と左手で 4.4cm 差)。"""
    import numpy as np

    from inference.desktop.perception.g1_urdf_fk import LEFT_WRIST_TOOL_OFFSET_M

    cfg = _load_skill_config(str(_VENDOR_DESKTOP))
    override = cfg["skills"]["rotate_table_base"]["wrist_tool_offset"]["left"]

    assert not np.allclose(override, LEFT_WRIST_TOOL_OFFSET_M), (
        "override が既定と同じなら、この test は何も守っていない"
    )

    skill = _build("rotate_table_base", "RotateTableBaseVlaSkill", "groot_overlay").skill
    np.testing.assert_allclose(skill._fk._left_offset, override)


def test_the_transition_graph_loops_over_four_legs():
    """脚を 4 本まわす。終端にすると 1 本で止まる。"""
    assert _TRANSITIONS["rotate_leg_to_tighten"] == ["rotate_table_base"]
    assert _LEGS == 4

    # 列が 1 本の輪になっていること (どの skill からも次が 1 つ)
    for skill_name, _cls in _STAGE_SKILLS:
        assert len(_TRANSITIONS[skill_name]) == 1, skill_name


def test_the_loop_increments_the_leg_counter():
    """`rotate_leg_to_tighten` から戻ると n_legs_completed が上がること。

    `SkillState.transition()` の副作用に依存しているので、graph を変えたときに
    ここが壊れていないか見る。
    """
    from inference.desktop.skill_planner.state import SkillState

    state = SkillState(current_skill="rotate_leg_to_tighten", n_legs_completed=0)
    state.transition("rotate_table_base")

    assert state.n_legs_completed == 1


# ---------------------------------------------------------------- variant の正本
#
# 2026-09-21 まで、submit の driver が `groot_pick_legs_v2` を直接書いていて、
# 本体 config の `default_variant_by_skill` は `groot_pick_legs_v1` だった。
# **大会経路と自前経路が別の ckpt で走っていた。** 二重管理をやめた再発防止。
def test_the_stage_variants_come_from_the_config_not_from_this_module():
    """driver は variant を持たず、config の `default_variant_by_skill` を読むこと。"""
    import components.ramen.orchestrator_driver as drv

    for entry in _STAGE_SKILLS:
        assert len(entry) == 2, (
            f"_STAGE_SKILLS に variant が書かれている: {entry}. "
            "正本は policy_config.yaml の default_variant_by_skill"
        )

    source = Path(drv.__file__).read_text(encoding="utf-8")
    # 実 ckpt 名が module に散らばっていないこと (docstring も含めて禁止)。
    for banned in ("groot_pick_legs_v1", "groot_pick_legs_v2", "groot_insert_leg_200k"):
        assert banned not in source, f"{banned} が driver に直書きされている"


def test_every_stage_skill_has_a_config_default():
    """4 skill 全部に既定があること (欠けたら起動時に落ちる)。"""
    defaults = _stage_variants(_POLICY_CFG)

    assert set(defaults) >= {name for name, _cls in _STAGE_SKILLS}


def test_the_pick_default_is_the_one_the_self_path_uses():
    """pick は本体 config が指す版で走ること。

    自前経路 (`entrypoint.fill_policy_variants_from_config`) と同じ節を読んでいる
    ので、この 2 つが食い違ったら vendor 同期が漏れている。
    """
    assert _stage_variants(_POLICY_CFG)["pick_table_leg"] == "groot_pick_legs_v1"
