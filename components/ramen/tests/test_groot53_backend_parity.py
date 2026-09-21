"""`RAMEN_POLICY=groot_53d_real` が自前経路と同じ prompt / 腰の扱いになること。

GPU / model 不要 (ヘルパーだけを見る)。

## なぜ必要か

`Groot53Backend` は VlaSkill を通さず `policy.predict` を直接呼ぶ薄い経路なので、
VlaSkill が持っている 2 つの規約が抜けていた (2026-09-21):

  prompt          `self._language_override or self.LANGUAGE` で決まるはずが、
                  運営から来る `obs["prompt"]` 任せだった。model が学習した
                  task 文字列と違うもので推論することになる
  dispatch_waist  `rotate_table_base` と `flip_table` は config で False なのに、
                  model の waist をそのまま (T,25) の腰列に載せていた

**flip は Stage 5 の唯一の skill で、大会経路ではこの backend でしか動かせない。**
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_VENDOR_DESKTOP = _HERE.parent / "vendor" / "desktop"
for _p in (str(_VENDOR_DESKTOP), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components.ramen.policy import (  # noqa: E402
    _VARIANT_SKILL,
    _variant_dispatch_waist,
    _variant_task,
)
from inference.desktop.lower_policy.policies.config_loader import (  # noqa: E402
    load_policy_variant,
)
from inference.desktop.lower_policy.skills import vla_skill as _vla  # noqa: E402

_REPO = str(_VENDOR_DESKTOP)
_POLICY_CFG = str(
    _VENDOR_DESKTOP / "inference/desktop/lower_policy/configs/policy_config.yaml"
)


def _task(variant: str) -> str | None:
    return _variant_task(_REPO, variant, load_policy_variant(_POLICY_CFG, variant))


@pytest.mark.parametrize("variant", sorted(_VARIANT_SKILL))
def test_every_mapped_variant_resolves_a_task(variant: str):
    """prompt が必ず決まること。None だと運営の prompt に落ちる。"""
    assert _task(variant), variant


def test_flip_uses_the_skill_language():
    """flip は variant に override が無いので class の LANGUAGE を使う。"""
    assert _task("groot_flip_table_n17_2_baseline") == _vla.FlipTableVlaSkill.LANGUAGE
    assert _task("groot_flip_table_n17_2_baseline") == "flip table"


def test_rotate_table_base_uses_the_variant_override():
    """variant の override が class の既定より優先されること。"""
    assert _task("groot_overlay") == "rotate table base"
    assert _task("groot_overlay") != _vla.RotateTableBaseVlaSkill.LANGUAGE


def test_dispatch_waist_matches_the_skill_config():
    """腰を出してよい skill だけ True。"""
    assert _variant_dispatch_waist(_REPO, "groot_overlay") is False
    assert _variant_dispatch_waist(_REPO, "groot_flip_table_n17_2_baseline") is False
    assert _variant_dispatch_waist(_REPO, "groot_insert_leg_200k") is True
    assert _variant_dispatch_waist(_REPO, "groot_rotate_leg_200k") is True


def test_unknown_variant_keeps_the_previous_behaviour():
    """表に無い variant は従来どおり (prompt は obs 任せ、腰は出す)。"""
    # override を持たない entry を使う (持っていると表を見る前に override が返る)。
    entry = load_policy_variant(_POLICY_CFG, "groot_insert_leg_200k")
    assert entry.policy_config.language_prompt is None
    assert _variant_task(_REPO, "not_a_variant", entry) is None
    assert _variant_dispatch_waist(_REPO, "not_a_variant") is True


def test_the_mapping_covers_every_53d_variant_the_server_exposes():
    """server が案内する 53D variant が表から漏れていないこと。"""
    from components.ramen.policy import _SKILL_BACKENDS  # noqa: F401

    for variant in (
        "groot_overlay",
        "groot_insert_leg_200k",
        "groot_rotate_leg_200k",
        "groot_flip_table_n17_2_baseline",
    ):
        assert variant in _VARIANT_SKILL, variant
