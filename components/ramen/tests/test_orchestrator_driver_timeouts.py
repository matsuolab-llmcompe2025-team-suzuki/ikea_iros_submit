"""boundary の driver が timeout の受け皿を持っていることを確認する。

GPU / YOLO / model 不要。`_load_stage_timeouts` と、driver が
`Orchestrator` に何を渡すかだけを見る。

## なぜ必要か

`Orchestrator.tick()` は **YOLO の `enter_check` しか見ない**。
`is_complete` / `max_seconds_hard` / `max_dwell_sec` は
`advance_finished_skill()` 側にあり、自前経路 (`run_live`) はそれを呼ぶが、
boundary の driver は `tick()` を直接回していたので **受け皿が無かった**。

会場は学習データと違うシーン (CLAUDE.md) なので、照明・遮蔽・画角で YOLO が
落とせば skill が進まないまま stage が終わる。実 image を載せた pod で
`rotate_table_base` から遷移しないことを実測している (2026-09-21)。
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

from components.ramen.orchestrator_driver import _STAGE_SKILLS, _load_stage_timeouts


def test_every_stage_skill_has_a_hard_timeout():
    """4 skill 全てに受け皿があること。1 つでも欠けるとそこで止まる。"""
    hard, actions = _load_stage_timeouts(str(_VENDOR_DESKTOP))

    missing = [name for name, _cls in _STAGE_SKILLS if name not in hard]
    assert missing == [], f"timeout が無い skill: {missing}"
    for name in hard:
        assert hard[name] > 0, f"{name}: {hard[name]}"
        assert actions[name] in ("advance", "stop"), f"{name}: {actions[name]}"


def test_the_timeout_seconds_match_the_skill_config_yaml():
    """秒数は driver が独自に持たず、自前経路と同じ YAML から来ていること。

    ⚠️ `on_timeout` の **action だけ** は意図的に YAML と違う。大会経路は
    `advance` に倒している (理由は `_load_stage_timeouts` の docstring と
    下の test)。秒数まで独自に持つと、自前経路と別物の調整になってしまう。
    """
    import yaml

    cfg = _VENDOR_DESKTOP / "inference/desktop/lower_policy/configs/skill_config.yaml"
    skills = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("skills") or {}

    hard, _actions = _load_stage_timeouts(str(_VENDOR_DESKTOP))

    for name, timeout_s in hard.items():
        assert timeout_s == float(skills[name]["max_seconds_hard"])


def test_rejects_a_non_positive_timeout(tmp_path, monkeypatch):
    """YAML が壊れていたら黙って 0 秒で回さず落ちること。"""
    fake = tmp_path / "inference/desktop/lower_policy/configs"
    fake.mkdir(parents=True)
    (fake / "skill_config.yaml").write_text(
        "skills:\n  rotate_table_base:\n    max_seconds_hard: 0\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="max_seconds_hard"):
        _load_stage_timeouts(str(tmp_path))


def test_the_orchestrator_accepts_the_maps_the_driver_builds():
    """Orchestrator 側の検証 (正の秒数 / 既知の action) を実際に通すこと。"""
    from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
    from inference.desktop.lower_policy.skills.mock import MockSkill
    from inference.desktop.orchestrator import Orchestrator
    from inference.desktop.perception.stream import DetectionStream

    hard, actions = _load_stage_timeouts(str(_VENDOR_DESKTOP))
    registry = {name: MockSkill(name) for name, _cls in _STAGE_SKILLS}

    orch = Orchestrator(
        _StubPerception(),
        DetectionStream(
            {
                "max_count": {"table_top": 1},
                "over_max_continue_iou": 0.3,
                "under_max_similar_iou": 0.3,
                "median_filter": {"enabled": False, "iou_match_min": 0.5},
            }
        ),
        SkillDispatchLowerPolicy(registry),
        initial_skill="rotate_table_base",
        transitions={name: [] for name, _c in _STAGE_SKILLS},
        enter_check={},
        hard_timeout_by_skill=hard,
        timeout_action_by_skill=actions,
    )

    assert orch.hard_timeout_by_skill == hard
    assert orch.timeout_action_by_skill == actions


class _StubPerception:
    def predict(self, rgb):  # noqa: ANN001
        return []


# ---------------------------------------------------------------- 大会経路の on_timeout
#
# #148 が YAML の `on_timeout` を全 skill `advance` -> `stop` に変えた。あれは
# **実機 SDK 経路** の判断 (空の腕のまま insert へ進んで卓にぶつからない)。
# 大会経路は前提が違うので、driver 側で `advance` に倒している。
def test_the_boundary_path_advances_on_timeout_regardless_of_the_yaml(monkeypatch):
    """YAML が stop でも、大会経路の既定は advance であること。

    脚の 4 skill には他の受け皿が無い (`is_complete` は常に False、
    `max_dwell_sec` は move_to_table だけ)。ここが stop だと **YOLO が外した
    時点でそのエピソードは何も進まないまま終わる**。
    """
    monkeypatch.delenv("RAMEN_ON_TIMEOUT", raising=False)

    hard, actions = _load_stage_timeouts(str(_VENDOR_DESKTOP))

    assert set(actions) == set(hard)
    assert set(actions.values()) == {"advance"}


def test_the_venue_can_switch_back_to_stop(monkeypatch):
    """会場で危ないと判断したら env で戻せること。"""
    monkeypatch.setenv("RAMEN_ON_TIMEOUT", "stop")

    _hard, actions = _load_stage_timeouts(str(_VENDOR_DESKTOP))

    assert set(actions.values()) == {"stop"}


def test_an_unknown_timeout_action_is_rejected(monkeypatch):
    """typo で黙って既定に落ちないこと。"""
    monkeypatch.setenv("RAMEN_ON_TIMEOUT", "halt")

    with pytest.raises(ValueError, match="RAMEN_ON_TIMEOUT"):
        _load_stage_timeouts(str(_VENDOR_DESKTOP))


def test_the_seconds_still_come_from_the_yaml(monkeypatch):
    """秒数は YAML のまま (action だけを大会経路用に倒している)。"""
    monkeypatch.delenv("RAMEN_ON_TIMEOUT", raising=False)

    hard, _actions = _load_stage_timeouts(str(_VENDOR_DESKTOP))

    assert hard["rotate_table_base"] == 30.0
    assert hard["pick_table_leg"] == 21.0
    assert hard["insert_table_leg"] == 21.0
    assert hard["rotate_leg_to_tighten"] == 58.0
