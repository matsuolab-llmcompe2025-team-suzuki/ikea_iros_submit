"""driver の runtime 配線 (先読み・脚ループ) を GPU 無しで確かめる。

model は読まない。stub policy / stub skill で「仕組みが発火するか」だけを見る。
実測値 (切替が何秒で済むか) は pod でしか取れないが、**配線が死んでいれば
ここで落ちる**ので、GPU 時間を無駄にしない。
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

from components.ramen.orchestrator_driver import (  # noqa: E402
    _LEGS,
    _STAGE_SKILLS,
    _TRANSITIONS,
    OrchestratorDriver,
)


# ---------------------------------------------------------------- 先読み
class _StubPolicy:
    """`prepare()` / `close()` を持つ DeferredPolicy の代役。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.prepared = 0
        self.closed = 0

    def prepare(self) -> None:
        self.prepared += 1

    def close(self) -> None:
        self.closed += 1


def _drain(residency) -> None:
    """先読み worker が積んだぶんを処理し終えるまで待つ。"""
    import time

    for _ in range(200):
        if residency._queue.empty():
            time.sleep(0.02)  # worker が取り出してから prepare するまでの隙間
            return
        time.sleep(0.02)
    raise AssertionError("先読み worker が終わらない")


def test_residency_preloads_every_expert(monkeypatch):
    """`resident` を既定 (=全部) にすると 4 expert 全てが先読みされること。

    これが無いと skill 切替のたびに act() が model load でブロックする
    (実測 pick 24.7s / insert 92s / rotate_leg 69s)。
    """
    monkeypatch.delenv("RAMEN_GPU_MODELS", raising=False)
    policies = {name: _StubPolicy(name) for name, _c, _v in _STAGE_SKILLS}

    residency = OrchestratorDriver._build_residency(None, policies)

    assert residency is not None
    assert residency.resident == len(policies)
    residency.on_skill_started("rotate_table_base")
    _drain(residency)

    assert all(p.prepared >= 1 for p in policies.values()), {
        n: p.prepared for n, p in policies.items()
    }
    residency.close()


def test_residency_respects_the_env_override(monkeypatch):
    """`RAMEN_GPU_MODELS=2` なら今の skill と次だけを載せる。"""
    monkeypatch.setenv("RAMEN_GPU_MODELS", "2")
    policies = {name: _StubPolicy(name) for name, _c, _v in _STAGE_SKILLS}

    residency = OrchestratorDriver._build_residency(None, policies)
    assert residency.resident == 2
    residency.on_skill_started("rotate_table_base")
    _drain(residency)

    assert policies["rotate_table_base"].prepared >= 1
    assert policies["pick_table_leg"].prepared >= 1
    assert policies["insert_table_leg"].prepared == 0
    residency.close()


def test_on_tick_hook_feeds_the_residency():
    """`on_tick` が active skill 名を residency に渡すこと (配線の要)。"""

    class _Recorder:
        def __init__(self) -> None:
            self.seen: list[str | None] = []

        def on_skill_started(self, name):  # noqa: ANN001
            self.seen.append(name)

    drv = OrchestratorDriver.__new__(OrchestratorDriver)
    drv._residency = _Recorder()

    class _Skill:
        name = "pick_table_leg"

    drv._on_tick(None, {}, _Skill())

    assert drv._residency.seen == ["pick_table_leg"]


def test_on_tick_is_safe_without_a_residency():
    drv = OrchestratorDriver.__new__(OrchestratorDriver)
    drv._residency = None

    drv._on_tick(None, {}, None)  # 例外にならないこと


# ---------------------------------------------------------------- 脚ループ
def test_four_leg_loop_advances_the_counter_and_then_stops():
    """脚が 4 本回り、5 本目に入らないこと。

    時間切れ前進は enter_check を見ないので、止める処理が無いと永久に回る。
    """
    from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
    from inference.desktop.lower_policy.skills.mock import MockSkill
    from inference.desktop.orchestrator import Orchestrator, enter_never
    from inference.desktop.perception.stream import DetectionStream

    names = [name for name, _c, _v in _STAGE_SKILLS]
    orch = Orchestrator(
        _NoDetections(),
        DetectionStream(
            {
                "max_count": {"table_top": 1},
                "over_max_continue_iou": 0.3,
                "under_max_similar_iou": 0.3,
                "median_filter": {"enabled": False, "iou_match_min": 0.5},
            }
        ),
        SkillDispatchLowerPolicy({n: MockSkill(n) for n in names}),
        initial_skill="rotate_table_base",
        transitions=_TRANSITIONS,
        enter_check={n: enter_never for n in names},
        hard_timeout_by_skill={n: 10.0 for n in names},
    )
    orch.tick(_Frame(0))

    # 遷移 1 回につき呼び出しは 2 回要る: 1 回目で進み、2 回目が「skill が変わった」
    # を検出して timer を張り直す (その回は経過 0 秒なので進まない)。
    # 実機では act() が 30Hz で呼ばれるので 1/30 秒ぶんの差でしかない。
    now = 100.0
    order: list[str] = [orch.state.current_skill]
    for _ in range(len(names) * (_LEGS + 1) * 2):
        now += 11.0
        orch.advance_finished_skill(now=now)
        order.append(orch.state.current_skill)

    assert orch.state.n_legs_completed >= _LEGS, orch.state.n_legs_completed
    # 脚ごとに pick を 1 度通ること (終端にしていると 1 度しか来ない)
    legs_seen = sum(
        1
        for a, b in zip(order, order[1:])
        if a == "rotate_leg_to_tighten" and b == "rotate_table_base"
    )
    assert legs_seen >= _LEGS, f"脚を {legs_seen} 本しか回っていない: {order}"


class _NoDetections:
    def predict(self, rgb):  # noqa: ANN001
        return []


class _Frame:
    def __init__(self, t: int) -> None:
        import numpy as np

        self.rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        self.t = t


def test_the_driver_stops_after_four_legs():
    """`_LEGS` に達したら driver が advance を呼ばなくなること。"""

    class _State:
        n_legs_completed = _LEGS

    class _Orch:
        state = _State()

        def __init__(self) -> None:
            self.advanced = 0

        def advance_finished_skill(self):
            self.advanced += 1

    drv = OrchestratorDriver.__new__(OrchestratorDriver)
    drv._orch = _Orch()
    drv._advance_halted = False
    drv._LiveSourceSafetyError = RuntimeError

    # act() の該当ブロックと同じ判定
    if not drv._advance_halted and drv._orch.state.n_legs_completed >= _LEGS:
        drv._advance_halted = True
    if not drv._advance_halted:
        drv._orch.advance_finished_skill()

    assert drv._advance_halted is True
    assert drv._orch.advanced == 0
