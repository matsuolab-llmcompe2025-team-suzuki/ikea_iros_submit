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


def _build(skill_name: str, cls_name: str, variant: str):
    """driver と同じ手順で 1 skill を組む。"""
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
    for skill_name, cls_name, variant in _STAGE_SKILLS:
        skill = _build(skill_name, cls_name, variant).skill
        assert getattr(skill, "_motion_limiter", None) is not None, skill_name
        assert getattr(skill, "_teacher_range", None) is not None, skill_name
        assert getattr(skill, "_fk", None) is not None, skill_name


def test_dispatch_waist_comes_from_the_skill_config():
    """`rotate_table_base` は False。True 固定だと腰を出してはいけない skill で出す。"""
    by_name = {
        name: _build(name, cls, variant).dispatch_waist
        for name, cls, variant in _STAGE_SKILLS
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
    for skill_name, _cls, _v in _STAGE_SKILLS:
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
