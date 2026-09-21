"""`RAMEN_PICK_HYBRID` で pick を hybrid に差し替える配線 (Issue #148)。

会場では **既定が今までどおりの GR00T** であることが一番大事なので、
「env を付けないと何も変わらない」を最初に確かめる。hybrid 側は VLM を
実際に立てずに確かめたいので、endpoint の probe だけ差し替える。

model は読まない (deferred のまま)。ここで見るのは配線だけ。
"""

from __future__ import annotations

import socket
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
    OrchestratorDriver,
    _hybrid_pick_settings,
    _load_skill_config,
)

_HYBRID_ENV = (
    "RAMEN_PICK_HYBRID",
    "RAMEN_PICK_HYBRID_CONFIG",
    "RAMEN_PICK_VLM_ENDPOINT",
    "RAMEN_PICK_VLM_MODEL",
)


@pytest.fixture
def clean_env(monkeypatch):
    """hybrid 系の env を全部落とす (開発機の設定が漏れ込まないように)。"""
    for key in _HYBRID_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.fixture
def stub_probe(monkeypatch):
    """VLM を立てずに済ませる。probe が呼ばれた回数を返す。"""
    import inference.desktop.pick_leg_hybrid.real_skill as real_skill

    calls: list = []

    def _probe(cfg, references=None, timeout_sec=3.0):  # noqa: ANN001
        calls.append((cfg.vlm.endpoint, cfg.vlm.model, len(references or [])))
        return {cfg.vlm.model}

    monkeypatch.setattr(real_skill, "probe_vlm_endpoint", _probe)
    return calls


def _skill_config() -> dict:
    return _load_skill_config(str(_VENDOR_DESKTOP))


# ---------------------------------------------------------------- 既定は変わらない
def test_without_the_env_nothing_changes(clean_env):
    """env が無ければ hybrid の材料を作らない = pick は GR00T のまま。

    会場の既定はこちら。**この test が落ちたら、env を付けていない運用が
    黙って別物になっている。**
    """
    cfg = _skill_config()
    before = dict(cfg["skills"]["pick_table_leg"])

    assert _hybrid_pick_settings(cfg) is None

    # skill_config を書き換えていないこと (dispatch_waist を勝手に潰さない)。
    assert cfg["skills"]["pick_table_leg"] == before


def test_the_built_driver_uses_the_groot_pick_by_default(clean_env, monkeypatch):
    """組み立てまで通して、pick が既存 class のままであること。"""
    _stub_yolo(monkeypatch)
    from inference.desktop.lower_policy.skills.vla_skill import PickTableLegVlaSkill

    drv = OrchestratorDriver()
    try:
        pick = drv._orch.dispatcher._skills["pick_table_leg"]
        assert type(pick) is PickTableLegVlaSkill
    finally:
        drv.close()


# ---------------------------------------------------------------- env を付けたとき
def test_the_env_selects_the_hybrid_class(clean_env, stub_probe):
    """`RAMEN_PICK_HYBRID=1` で hybrid の class と kwargs が返ること。"""
    from inference.desktop.pick_leg_hybrid.real_skill import RealPickLegHybridVlaSkill

    clean_env.setenv("RAMEN_PICK_HYBRID", "1")
    cfg = _skill_config()

    settings = _hybrid_pick_settings(cfg)
    assert settings is not None

    assert settings.skill_cls is RealPickLegHybridVlaSkill
    assert len(stub_probe) == 1, "endpoint を組み立て前に probe していない"
    # 参照画像 2 枚を渡して probe していること。
    assert stub_probe[0][2] == 2


def test_the_hybrid_never_dispatches_waist(clean_env, stub_probe):
    """腰は Regular Mode に残す。

    `build_vla_skill` が読む **前に** 潰しておかないと、MotionLimiter の包絡と
    `BuiltSkill` の契約が食い違う (自前経路 entrypoint.py と同じ順序)。
    """
    clean_env.setenv("RAMEN_PICK_HYBRID", "1")
    cfg = _skill_config()

    _hybrid_pick_settings(cfg)

    assert cfg["skills"]["pick_table_leg"]["dispatch_waist"] is False


def test_phase3_is_pinned_to_the_rule_based_executor(clean_env, stub_probe):
    """区間 3 は rule_based 固定。

    VLA 版には有限の完了条件が無く、自前経路の `_validate_phase3_config` も
    本番では rule_based しか許さない。driver 側で選べるようにしない。
    """
    clean_env.setenv("RAMEN_PICK_HYBRID", "1")
    settings = _hybrid_pick_settings(_skill_config())

    assert settings.extra_kwargs["phase3_executor"] == "rule_based"


def test_the_handover_target_comes_from_the_canonical_insert_pose(
    clean_env, stub_probe
):
    """区間 3 の終点は skill_config.yaml の insert frame-0 (正本) から取ること。"""
    from inference.desktop.lower_policy.initial_pose import initial_pose_from_config

    clean_env.setenv("RAMEN_PICK_HYBRID", "1")
    cfg = _skill_config()
    extra = _hybrid_pick_settings(cfg).extra_kwargs

    expected = initial_pose_from_config(cfg, "insert_table_leg")
    assert extra["next_initial_arm_target"].tolist() == (
        expected.arm_position_rad.tolist()
    )
    # dex1_target_rad は tuple (自前経路 entrypoint.py が渡すのと同じ形)。
    assert tuple(extra["next_initial_hand_target"]) == tuple(expected.dex1_target_rad)


def test_the_hybrid_owns_the_pick_timeout(clean_env, stub_probe, monkeypatch):
    """pick の時間切れが hybrid 自身の予算 (30s / stop) で上書きされること。

    skill_config の 21s は従来 pick 用で VLM の揺らぎを見込んでいない。
    上書きしないと **hybrid が自分の予算を使い切る前に切られる**。
    自前経路 (`entrypoint.py`) も同じ上書きをしている。
    """
    _stub_yolo(monkeypatch)
    clean_env.setenv("RAMEN_PICK_HYBRID", "1")

    drv = OrchestratorDriver()
    try:
        assert drv._orch.hard_timeout_by_skill["pick_table_leg"] == 30.0
        assert drv._orch.timeout_action_by_skill["pick_table_leg"] == "stop"
        # 他の skill は YAML のまま (hybrid は pick にしか効かない)。
        assert drv._orch.hard_timeout_by_skill["insert_table_leg"] == 21.0
    finally:
        drv.close()


def test_without_the_hybrid_the_pick_timeout_stays_at_the_yaml_value(
    clean_env, monkeypatch
):
    """既定では YAML の 21s のまま = 今までどおり。"""
    _stub_yolo(monkeypatch)

    drv = OrchestratorDriver()
    try:
        assert drv._orch.hard_timeout_by_skill["pick_table_leg"] == 21.0
    finally:
        drv.close()


def test_the_endpoint_and_model_can_be_overridden(clean_env, stub_probe):
    """会場で VLM を別ホストに立てても焼き直さずに済むこと。"""
    clean_env.setenv("RAMEN_PICK_HYBRID", "1")
    clean_env.setenv(
        "RAMEN_PICK_VLM_ENDPOINT", "http://10.0.0.5:9000/v1/chat/completions"
    )
    clean_env.setenv("RAMEN_PICK_VLM_MODEL", "some/other-vlm")

    extra = _hybrid_pick_settings(_skill_config()).extra_kwargs

    assert stub_probe[0][0] == "http://10.0.0.5:9000/v1/chat/completions"
    assert stub_probe[0][1] == "some/other-vlm"
    assert extra["hybrid_vlm_endpoint"] == "http://10.0.0.5:9000/v1/chat/completions"
    assert extra["hybrid_vlm_model"] == "some/other-vlm"


# ---------------------------------------------------------------- 落ちるべきところで落ちる
def test_a_missing_vlm_fails_at_startup_not_mid_episode(clean_env):
    """VLM が居ないなら **組み立ての時点で** 落ちること。

    ここで落ちないと、気付けるのは `pick_table_leg` に入った後になる。
    hybrid は境界が立たないまま `hard_timeout_sec` (30s) で HOLD 停止するので、
    会場では「掴まない」としか見えず、原因に辿り着けない。
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    # with を抜けた時点で誰も listen していない port。

    clean_env.setenv("RAMEN_PICK_HYBRID", "1")
    clean_env.setenv(
        "RAMEN_PICK_VLM_ENDPOINT",
        f"http://127.0.0.1:{dead_port}/v1/chat/completions",
    )

    with pytest.raises(RuntimeError, match="not reachable"):
        _hybrid_pick_settings(_skill_config())


def test_the_flag_only_accepts_explicit_truthy_values(clean_env, stub_probe):
    """`RAMEN_PICK_HYBRID=0` や空文字で誤って有効にならないこと。"""
    cfg = _skill_config()
    for value in ("", "0", "false", "off", "no"):
        clean_env.setenv("RAMEN_PICK_HYBRID", value)
        assert _hybrid_pick_settings(cfg) is None, value
    for value in ("1", "true", "TRUE", "yes", "on"):
        clean_env.setenv("RAMEN_PICK_HYBRID", value)
        assert _hybrid_pick_settings(cfg) is not None, value


# ---------------------------------------------------------------- 組み立てまで通す
def test_only_pick_is_replaced(clean_env, stub_probe, monkeypatch):
    """hybrid を有効にしても、差し替わるのは pick だけであること。"""
    _stub_yolo(monkeypatch)
    from inference.desktop.lower_policy.skills.vla_skill import (
        InsertTableLegVlaSkill,
        RotateLegToTightenVlaSkill,
        RotateTableBaseVlaSkill,
    )
    from inference.desktop.pick_leg_hybrid.real_skill import RealPickLegHybridVlaSkill

    clean_env.setenv("RAMEN_PICK_HYBRID", "1")
    drv = OrchestratorDriver()
    try:
        skills = drv._orch.dispatcher._skills
        assert isinstance(skills["pick_table_leg"], RealPickLegHybridVlaSkill)
        assert type(skills["rotate_table_base"]) is RotateTableBaseVlaSkill
        assert type(skills["insert_table_leg"]) is InsertTableLegVlaSkill
        assert type(skills["rotate_leg_to_tighten"]) is RotateLegToTightenVlaSkill
    finally:
        drv.close()


def _stub_yolo(monkeypatch) -> None:
    """YOLO の重みを読まずに driver を組む (test_..._runtime.py と同じ手)。"""
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
    monkeypatch.delenv("RAMEN_GPU_MODELS", raising=False)
    monkeypatch.delenv("RAMEN_ORCH_LOG", raising=False)
