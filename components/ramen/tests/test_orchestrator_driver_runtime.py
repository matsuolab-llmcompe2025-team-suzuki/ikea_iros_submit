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
    _INITIAL_SKILL,
    _LEGS,
    _STAGE_SKILLS,
    _TRANSITIONS,
    OrchestratorDriver,
)


def _driver_stub(start_skill: str | None = None):
    """`_build_residency` だけを呼ぶための最小の self。

    `_resume` (どの脚から始めるか) を見るようになったので、素の None では呼べない。
    """
    from components.ramen.orchestrator_driver import _LEGS as _L
    from components.ramen.orchestrator_driver import _Resume

    stub = OrchestratorDriver.__new__(OrchestratorDriver)
    stub._resume = _Resume(
        start_skill=start_skill or _INITIAL_SKILL, legs_done=0, end_legs=_L
    )
    return stub


# ---------------------------------------------------------------- 先読み
class _StubPolicy:
    """`prepare()` / `close()` を持つ DeferredPolicy の代役。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.prepared = 0
        self.closed = 0
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def prepare(self) -> None:
        self.prepared += 1
        self.loaded = True

    def close(self) -> None:
        self.closed += 1
        self.loaded = False


def _drain(residency) -> None:
    """先読み worker が積んだぶんを処理し終えるまで待つ。"""
    import time

    for _ in range(200):
        if residency._queue.empty():
            time.sleep(0.02)  # worker が取り出してから prepare するまでの隙間
            return
        time.sleep(0.02)
    raise AssertionError("先読み worker が終わらない")


def test_residency_preloads_the_next_expert(monkeypatch):
    """既定で「今 + 次」が載ること。

    これが無いと skill 切替のたびに act() が model load でブロックする
    (実測 pick 24.7s / insert 92s / rotate_leg 69s)。全部載せる必要は無い:
    読み込み 約 8 秒 < 各 skill の 21〜58 秒 なので、次の 1 つで間に合う。
    """
    monkeypatch.delenv("RAMEN_GPU_MODELS", raising=False)
    policies = {name: _StubPolicy(name) for name, _c, _v in _STAGE_SKILLS}

    residency = OrchestratorDriver._build_residency(_driver_stub(), policies)

    assert residency is not None
    assert residency.resident == 2  # 全部載せると 53D×4 + pick で約 26 GiB
    residency.on_skill_started("rotate_table_base")
    _drain(residency)

    try:
        assert policies["rotate_table_base"].prepared >= 1
        assert policies["pick_table_leg"].prepared >= 1, "次が先読みされていない"
        assert policies["insert_table_leg"].prepared == 0, "2 つより多く載せている"
    finally:
        residency.close()


def test_residency_respects_the_env_override(monkeypatch):
    """`RAMEN_GPU_MODELS=2` なら今の skill と次だけを載せる。"""
    monkeypatch.setenv("RAMEN_GPU_MODELS", "2")
    policies = {name: _StubPolicy(name) for name, _c, _v in _STAGE_SKILLS}

    residency = OrchestratorDriver._build_residency(_driver_stub(), policies)
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


# ---------------------------------------------------- __init__ の配線
@pytest.fixture
def built_driver(monkeypatch):
    """YOLO だけ差し替えて driver を実際に構築する (model は deferred のまま)。

    `_build_residency` を直接呼ぶ test だけだと、`__init__` から配線が外れても
    気付けない。ここで組み立てまで通す。
    """
    import inference.desktop.perception.yolo_obb as yolo

    class _StubYolo:
        def __init__(self, *a, **k) -> None:
            pass

        def predict(self, rgb):  # noqa: ANN001
            return []

    monkeypatch.setattr(yolo, "YoloObbPerception", _StubYolo)
    monkeypatch.setattr(
        OrchestratorDriver, "_resolve_yolo_weight", staticmethod(lambda ref: "stub.pt")
    )
    for key in (
        "RAMEN_GPU_MODELS",
        "RAMEN_ORCH_LOG",
        "RAMEN_START_LEG",
        "RAMEN_END_LEG",
        "RAMEN_START_SKILL",
        "RAMEN_PICK_HYBRID",
    ):
        monkeypatch.delenv(key, raising=False)

    drv = OrchestratorDriver(prime_first_model=False)
    yield drv
    drv.close()


def test_the_driver_wires_the_residency_and_tick_hook(built_driver):
    """`__init__` が先読みと hook を Orchestrator に渡していること。"""
    assert built_driver._residency is not None, "先読みが配線から外れている"
    assert built_driver._orch.on_tick is not None, "on_tick が渡っていない"
    assert built_driver._orch.policy_filter is not None, "policy_filter が渡っていない"


def test_the_driver_passes_the_hard_timeouts(built_driver):
    """時間切れの受け皿が 4 skill 分渡っていること。"""
    hard = built_driver._orch.hard_timeout_by_skill
    for name, _cls, _variant in _STAGE_SKILLS:
        assert name in hard, name
        assert hard[name] > 0


def test_the_driver_uses_the_looping_transition_graph(built_driver):
    """脚が 1 本で終わらないこと。"""
    assert built_driver._orch.transitions["rotate_leg_to_tighten"] == [
        "rotate_table_base"
    ]


def test_the_log_sink_is_opt_in(monkeypatch, tmp_path):
    """`RAMEN_ORCH_LOG` が無ければ None、あれば書ける object。"""
    monkeypatch.delenv("RAMEN_ORCH_LOG", raising=False)
    assert OrchestratorDriver._build_log_sink() is None

    path = tmp_path / "orch.jsonl"
    monkeypatch.setenv("RAMEN_ORCH_LOG", str(path))
    sink = OrchestratorDriver._build_log_sink()
    try:
        assert sink is not None
        sink.write('{"probe": 1}\n')
    finally:
        sink.close()
    assert path.read_text(encoding="utf-8").strip() == '{"probe": 1}'


def test_residency_keeps_every_expert_at_every_point_of_the_leg(monkeypatch):
    """脚の末尾でも 4 つとも常駐し続けること。

    ModelResidency は order を **直線**として扱い `order[i:i+resident]` を保つ。
    1 脚ぶんの列だけを渡すと末尾 (`rotate_leg_to_tighten`、index 3) で keep が
    自分 1 つになり、他の 3 つを毎周回解放して次の脚で読み直す。
    実測では脚ごとに 8.5 秒のスパイクが出ていた (2026-09-21、pod)。
    """
    monkeypatch.delenv("RAMEN_GPU_MODELS", raising=False)
    policies = {name: _StubPolicy(name) for name, _c, _v in _STAGE_SKILLS}
    residency = OrchestratorDriver._build_residency(_driver_stub(), policies)

    try:
        for name in policies:
            index = residency._order.index(name)
            keep = set(residency._order[index : index + residency.resident])
            assert len(keep) == residency.resident, (
                f"{name} に居るとき keep が {sorted(keep)}"
            )
            assert name in keep
    finally:
        residency.close()


def test_the_next_skill_is_always_ready_before_it_starts(monkeypatch):
    """脚を 2 周しても、次に入る skill は毎回すでに読めていること。

    これが「切替が 8 秒止まらない」の中身。`resident=2` では窓から外れたものが
    解放されるのは **設計どおり**なので、守るべきは「解放されたかどうか」では
    なく「必要になった時点で載っているか」。
    """
    monkeypatch.delenv("RAMEN_GPU_MODELS", raising=False)
    names = [name for name, _c, _v in _STAGE_SKILLS]
    policies = {name: _StubPolicy(name) for name in names}
    residency = OrchestratorDriver._build_residency(_driver_stub(), policies)

    try:
        for leg in range(2):
            for i, name in enumerate(names):
                residency.on_skill_started(name)
                _drain(residency)
                nxt = names[(i + 1) % len(names)]
                assert policies[nxt].loaded, (
                    f"leg {leg}: {name} の次 ({nxt}) が読めていない"
                )
    finally:
        residency.close()


# ---------------------------------------------------- variant の差し替え
def test_variant_override_swaps_the_expert(monkeypatch):
    """`RAMEN_VARIANT_<SKILL>` で expert を入れ替えられること。

    会場で image を焼き直さずに rotate_table_base を RAMEN-Ori に振れるようにする。
    """
    from components.ramen.orchestrator_driver import _variant_override

    monkeypatch.delenv("RAMEN_VARIANT_ROTATE_TABLE_BASE", raising=False)
    assert _variant_override("rotate_table_base", "groot_overlay") == "groot_overlay"

    monkeypatch.setenv(
        "RAMEN_VARIANT_ROTATE_TABLE_BASE", "rotate_table_base_ramen_ori_141_c32"
    )
    assert (
        _variant_override("rotate_table_base", "groot_overlay")
        == "rotate_table_base_ramen_ori_141_c32"
    )


def test_the_override_target_exists_in_the_policy_config():
    """差し替え先として案内する variant が実在すること (typo 防止)。"""
    from inference.desktop.lower_policy.policies.config_loader import (
        load_policy_variant,
    )

    cfg = str(
        _VENDOR_DESKTOP / "inference/desktop/lower_policy/configs/policy_config.yaml"
    )
    entry = load_policy_variant(cfg, "rotate_table_base_ramen_ori_141_c32")

    assert entry.policy_type == "ramen_ori"
    assert entry.policy_config.ckpt_ref


def test_ramen_ori_brings_its_own_cameras_and_state_dim():
    """policy_type が違っても VlaSkill 側が policy に聞くので差し替えが効く。"""
    from inference.desktop.lower_policy.policies.groot import (
        CAMERAS as GROOT_CAMERAS,
    )
    from inference.desktop.lower_policy.policies.ramen_ori import RamenOriPolicy

    assert RamenOriPolicy.STATE_DIM == 71
    assert len(RamenOriPolicy.CAMERAS) == 4
    assert len(GROOT_CAMERAS) == 3
    assert callable(RamenOriPolicy.build_state_from_raw)


# ---------------------------------------------------- episode 間の reset
def test_reset_clears_the_halt_and_the_leg_counter(built_driver):
    """運営が episode 間に呼ぶ `reset` が停止状態を戻すこと。

    戻さないと 1 本走り切った後の 2 本目が skill を一切進めないまま終わる
    (`_advance_halted` が True のままで `advance_finished_skill` が呼ばれず、
    `n_legs_completed` も 4 のままで `enter_pick_table_leg` が常に False)。
    `reset` は boundary 契約の一部で `transport.py` が route している。
    """
    built_driver._advance_halted = True
    built_driver._orch.state.n_legs_completed = _LEGS
    built_driver._orch.state.transition("pick_table_leg")

    built_driver.reset()

    assert built_driver._advance_halted is False
    assert built_driver._orch.state.n_legs_completed == 0
    assert built_driver._orch.state.current_skill == _INITIAL_SKILL
    assert built_driver._orch.dispatcher.active_skill_name is None


def test_reset_keeps_the_models_resident(built_driver):
    """reset で model を解放しないこと (読み直すと切替と同じ待ちが出る)。"""
    before = built_driver._residency

    built_driver.reset()

    assert built_driver._residency is before


def test_enter_check_fails_loudly_on_an_unregistered_skill():
    """YOLO 判定を持たない skill は明示する。書き忘れを黙って通さない。"""
    from components.ramen.orchestrator_driver import _YOLO_FREE_ENTRY
    from inference.desktop.orchestrator import DEFAULT_ENTER_CHECK

    candidates = {c for cands in _TRANSITIONS.values() for c in cands}
    for c in candidates:
        assert c in _YOLO_FREE_ENTRY or c in DEFAULT_ENTER_CHECK, c
    # 明示リストは実在の遷移先だけであること (typo 検出)
    assert _YOLO_FREE_ENTRY <= candidates


# ---------------------------------------------------- 1 本目は卓を回さない
def test_the_first_leg_does_not_rotate_the_table(built_driver):
    """開始 skill が pick であること (自前経路の STAGE_SKILL_SEQUENCES[1] と同じ)。

    元は `rotate_table_base` 始まりで、4 脚に対して回転が 4 回あった。学習データの
    1 本目とは違う卓の向きから掴みにいくうえ、頭で 30 秒を余分に使う。
    """
    assert _INITIAL_SKILL == "pick_table_leg"
    assert built_driver._orch.state.current_skill == "pick_table_leg"


def test_the_loop_matches_the_self_path_stage_sequences():
    """ループを 1 周すると自前経路の stage 2..4 と同じ並びになること。"""
    from inference.desktop.orchestrator import STAGE_SKILL_SEQUENCES

    def _walk(start: str, steps: int) -> list[str]:
        path = [start]
        for _ in range(steps):
            path.append(_TRANSITIONS[path[-1]][0])
        return path

    # 1 本目: pick -> insert -> rotate_leg、その次が 2 本目の頭の rotate。
    first = _walk(_INITIAL_SKILL, len(STAGE_SKILL_SEQUENCES[1]) - 1)
    assert first == STAGE_SKILL_SEQUENCES[1]
    assert _TRANSITIONS[first[-1]][0] == "rotate_table_base"

    # 2 本目以降: rotate -> pick -> insert -> rotate_leg。
    second = _walk("rotate_table_base", len(STAGE_SKILL_SEQUENCES[2]) - 1)
    assert second == STAGE_SKILL_SEQUENCES[2]


def test_the_preload_order_starts_at_the_initial_skill(monkeypatch):
    """先読みの列が実際の実行順で始まること。

    `_STAGE_SKILLS` の並び (rotate_table_base 始まり) のままだと、構築時の
    `order[0:resident]` が「1 本目に使わない rotate」を読んで「次に要る insert」を
    読まない = 最初の切替で丸ごとブロックする。
    """
    monkeypatch.delenv("RAMEN_GPU_MODELS", raising=False)
    policies = {name: _StubPolicy(name) for name, _c, _v in _STAGE_SKILLS}

    residency = OrchestratorDriver._build_residency(_driver_stub(), policies)
    try:
        assert residency._order[0] == _INITIAL_SKILL
        assert residency._order[1] == _TRANSITIONS[_INITIAL_SKILL][0]
        # 4 脚ぶんに伸ばしてあること (1 脚だけだと末尾で毎周回解放する)
        assert len(residency._order) == len(_STAGE_SKILLS) * _LEGS
    finally:
        residency.close()


# ---------------------------------------------------- 途中の脚から再開する
#
# 自前経路の `--phase3-start-stage` / `--phase3-end-stage` に相当する。boundary の
# server は 1 本の process が走り続けるので stage を分けられないが、**会場で
# 3 本目からやり直せないと困る**ので同じ粒度を env で持たせている。
@pytest.fixture
def clean_resume_env(monkeypatch):
    for key in ("RAMEN_START_LEG", "RAMEN_END_LEG", "RAMEN_START_SKILL"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_the_default_is_the_whole_run_from_the_first_leg(clean_resume_env):
    from components.ramen.orchestrator_driver import _resume_settings

    resume = _resume_settings()

    assert resume.start_skill == _INITIAL_SKILL
    assert resume.legs_done == 0
    assert resume.end_legs == _LEGS


def test_starting_at_a_later_leg_rotates_the_table_first(clean_resume_env):
    """2 本目以降は卓を回してから pick (STAGE_SKILL_SEQUENCES[2] と同じ)。"""
    from components.ramen.orchestrator_driver import _resume_settings

    clean_resume_env.setenv("RAMEN_START_LEG", "3")
    resume = _resume_settings()

    assert resume.start_skill == "rotate_table_base"
    # 判定に効く数。0 だと `enter_pick_table_leg` が 1 本目の Kabsch に落ちる。
    assert resume.legs_done == 2
    assert resume.end_legs == _LEGS


def test_the_end_leg_can_stop_the_run_early(clean_resume_env):
    from components.ramen.orchestrator_driver import _resume_settings

    clean_resume_env.setenv("RAMEN_START_LEG", "2")
    clean_resume_env.setenv("RAMEN_END_LEG", "2")
    resume = _resume_settings()

    assert (resume.legs_done, resume.end_legs) == (1, 2)


def test_a_leg_inside_a_leg_can_be_resumed(clean_resume_env):
    """脚の途中 (insert から等) に戻す口。物理的な前提は運用側の責任。"""
    from components.ramen.orchestrator_driver import _resume_settings

    clean_resume_env.setenv("RAMEN_START_LEG", "2")
    clean_resume_env.setenv("RAMEN_START_SKILL", "insert_table_leg")
    resume = _resume_settings()

    assert resume.start_skill == "insert_table_leg"
    assert resume.legs_done == 1


@pytest.mark.parametrize(
    "env, match",
    [
        ({"RAMEN_START_LEG": "0"}, "RAMEN_START_LEG"),
        ({"RAMEN_START_LEG": "5"}, "RAMEN_START_LEG"),
        ({"RAMEN_START_LEG": "two"}, "RAMEN_START_LEG"),
        ({"RAMEN_END_LEG": "9"}, "RAMEN_END_LEG"),
        ({"RAMEN_START_LEG": "3", "RAMEN_END_LEG": "2"}, "RAMEN_END_LEG"),
        ({"RAMEN_START_SKILL": "flip_table"}, "RAMEN_START_SKILL"),
    ],
)
def test_bad_resume_settings_fail_at_startup(clean_resume_env, env, match):
    """typo で黙って 1 本目から回り直さないこと。"""
    from components.ramen.orchestrator_driver import _resume_settings

    for key, value in env.items():
        clean_resume_env.setenv(key, value)

    with pytest.raises(ValueError, match=match):
        _resume_settings()


def test_the_built_driver_resumes_at_the_requested_leg(monkeypatch):
    """組み立てまで通して、state と先読みの列が再開点に揃っていること。"""
    import inference.desktop.perception.yolo_obb as yolo

    class _StubYolo:
        def __init__(self, *a, **k) -> None:
            pass

        def predict(self, rgb):  # noqa: ANN001
            return []

    monkeypatch.setattr(yolo, "YoloObbPerception", _StubYolo)
    monkeypatch.setattr(
        OrchestratorDriver, "_resolve_yolo_weight", staticmethod(lambda ref: "stub.pt")
    )
    for key in ("RAMEN_GPU_MODELS", "RAMEN_ORCH_LOG", "RAMEN_START_SKILL",
                "RAMEN_END_LEG", "RAMEN_PICK_HYBRID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAMEN_START_LEG", "3")

    drv = OrchestratorDriver(prime_first_model=False)
    try:
        assert drv._orch.state.current_skill == "rotate_table_base"
        assert drv._orch.state.n_legs_completed == 2
        assert drv._residency._order[0] == "rotate_table_base"

        # reset は「1 本目から」ではなく **再開点** に戻すこと。戻さないと
        # reset のたびに enter_pick_table_leg が 1 本目の規則 (Kabsch) に落ちる。
        drv._advance_halted = True
        drv._orch.state.n_legs_completed = _LEGS
        drv.reset()
        assert drv._advance_halted is False
        assert drv._orch.state.n_legs_completed == 2
        assert drv._orch.state.current_skill == "rotate_table_base"
    finally:
        drv.close()
