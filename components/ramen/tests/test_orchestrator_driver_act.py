"""`act()` が毎 tick 何をするかを、自前経路の `run_live` と突き合わせて固定する。

2026-09-21 の監査で見つかった 5 件の再発防止。どれも「走り切ってしまう」ので
4 脚通しの実測では出ず、`act()` を 1 行ずつ読んで初めて見つかった:

  1. `tick()` が try の外 → 時間切れで止めても expert が動き続ける
  2. `_dex1_src.update()` を呼んでおらず hand state が常に全開
  3. 手が未 dispatch の tick に 0 を入れて **全閉**を publish する
  4. `tick()` の例外が素通りして client 接続が切れる
  5. `obs["t"]` を捨てて自前カウンタを使い、カメラ鮮度が発火しない

model は読まない。`Orchestrator` を差し替えて `act()` の配線だけを見る。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_VENDOR_DESKTOP = _HERE.parent / "vendor" / "desktop"
for _p in (str(_VENDOR_DESKTOP), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components.ramen.g1_urdf_fk import G1WristFK  # noqa: E402
from components.ramen.orchestrator_driver import (  # noqa: E402
    _INITIAL_SKILL,
    _LEGS,
    _Resume,
    OrchestratorDriver,
)
from components.ramen.orchestrator_io import (  # noqa: E402
    BoundaryJointStateSource,
    BoundaryWristSource,
    InterceptorActuator,
    MeasuredDex1StateSource,
)
from inference.desktop.perception.dex1_state_source import (  # noqa: E402
    SyntheticDex1StateSource,
)

# insert_table_leg の frame-0 (脚を握った状態)。全開 4.5 とは大きく違う。
_INSERT_HAND = (0.679, 0.421)


class _FakeState:
    def __init__(self) -> None:
        self.n_legs_completed = 0
        self.current_skill = _INITIAL_SKILL


class _FakeResult:
    def __init__(self, action, skill) -> None:
        self.action = action
        self.current_skill = skill


class _FakeOrch:
    """`tick` / `advance_finished_skill` だけを持つ最小の Orchestrator。"""

    def __init__(self) -> None:
        self.state = _FakeState()
        self.ticks = 0
        self.advances = 0
        self.frames: list = []
        self.tick_raises: BaseException | None = None
        self.advance_raises: BaseException | None = None
        self.action = np.full(14, 0.3, dtype=np.float64)
        self.log_sink = None

    def tick(self, frame):  # noqa: ANN001
        self.ticks += 1
        self.frames.append(frame)
        if self.tick_raises is not None:
            raise self.tick_raises
        return _FakeResult(self.action, self.state.current_skill)

    def advance_finished_skill(self):
        self.advances += 1
        if self.advance_raises is not None:
            raise self.advance_raises


class _Safety(RuntimeError):
    """`LiveSourceSafetyError` の代役。"""


def _driver(initial_hand2=_INSERT_HAND) -> OrchestratorDriver:
    """`act()` を呼べる最小の driver (model も YOLO も無し)。"""
    drv = OrchestratorDriver.__new__(OrchestratorDriver)
    drv._t = 0
    drv._advance_halted = False
    drv._hold_reason = None
    drv._last_step19 = None
    drv._last_actions = None
    drv._last_obs_t = None
    drv._frame_received_ns = 0
    drv._head_bgr = None
    drv._head_duplicate_mono = True
    drv._stereo_seen = False
    drv._head_mono = False
    drv._head_generation = 0
    drv._head_received_ns = 0
    drv._missing_images = {"head": 0, "wrist_l": 0, "wrist_r": 0}
    drv._resume = _Resume(_INITIAL_SKILL, 0, _LEGS)
    drv._initial_hand2 = tuple(float(v) for v in initial_hand2)
    drv._LiveSourceSafetyError = _Safety
    drv._joint_src = BoundaryJointStateSource()
    drv._wrist_l = BoundaryWristSource()
    drv._wrist_r = BoundaryWristSource()
    drv._arm = InterceptorActuator("arm")
    drv._waist = InterceptorActuator("waist")
    drv._hand = InterceptorActuator("hand")
    drv._dex1_synth = SyntheticDex1StateSource(
        drv._initial_hand2, command_source=drv._hand
    )
    drv._dex1_src = MeasuredDex1StateSource(fallback=drv._dex1_synth)
    drv._dex1_was_measured = False
    drv._hybrid_needs_measured_dex1 = False
    drv._fk = G1WristFK.from_urdf()
    drv._ee_frame_transform = None
    drv._orch = _FakeOrch()
    return drv


def _obs(t: float = 1.0, *, images: dict | None = None) -> dict:
    img = np.zeros((480, 640, 3), np.uint8)
    if images is None:
        images = {"ego_view": img, "left_wrist": img, "right_wrist": img}
    return {
        "images": images,
        "body_q": np.linspace(0.0, 0.28, 29).astype(np.float64),
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "prompt": "assemble the table",
        "t": t,
    }


# ---------------------------------------------------------------- 2) hand state
def test_hand_state_echoes_the_commands_instead_of_being_stuck_open():
    """握った指令を出したら state もそれを返すこと。

    以前は `BoundaryDex1StateSource((1.0, 1.0))` を作って `update()` を一度も
    呼んでおらず、**ずっと全開 4.5 rad** だった。`insert_table_leg` /
    `rotate_leg_to_tighten` は脚を握った状態が frame 0 なので、
    「握っているのに開いている」と毎 tick model に伝えていた。
    """
    drv = _driver()

    # まだ何も指令していない = 開始 skill の frame-0。
    assert drv._dex1_src.get().position_rad == pytest.approx(_INSERT_HAND)

    drv._hand.send_action(np.array([0.1, 0.2]))

    assert drv._dex1_src.get().position_rad == pytest.approx([0.1, 0.2])


def test_the_hand_state_is_not_reset_between_ticks():
    """tick をまたいでも直近の hand 指令が残ること (実 publisher と同じ)。"""
    drv = _driver()
    drv._hand.send_action(np.array([0.1, 0.2]))

    drv.act(_obs(t=1.0))
    drv.act(_obs(t=2.0))

    assert drv._hand.last == pytest.approx([0.1, 0.2])
    assert drv._dex1_src.get().position_rad == pytest.approx([0.1, 0.2])


# ---------------------------------------------------------------- 3) 19D の埋め方
def test_an_undispatched_hand_tick_does_not_publish_a_full_close():
    """手を出さない tick で **全閉**を出さないこと。

    `assemble_19d` が 0 埋めだと `dex1_model_to_taskspace(0.0)` = `+1.0` = 全閉。
    deferred model の load 直後など `step()` が None を返す tick で起きる。
    """
    from components.ramen.taskspace_adapter import dex1_model_to_taskspace

    drv = _driver()
    out = drv.act(_obs())  # hand は一度も dispatch されていない

    taskspace = np.asarray(out["actions"])
    left, right = taskspace[0, 0:2], taskspace[0, 2:4]
    expected = [dex1_model_to_taskspace(v) for v in _INSERT_HAND]

    assert float(left[0]) == pytest.approx(expected[0])
    assert float(right[0]) == pytest.approx(expected[1])
    # 全閉 (+1.0) になっていないこと。
    assert float(left[0]) < 1.0 and float(right[0]) < 1.0


def test_an_undispatched_waist_tick_uses_the_measured_angles():
    """腰を出さない skill (rotate_table_base) で 0 ではなく実測を使うこと。

    0 のまま FK に入れると、実際の腰角と違う姿勢で EE を計算して運営へ渡す。
    """
    drv = _driver()
    obs = _obs()
    drv.act(obs)

    assert drv._last_step19 is not None
    assert drv._last_step19[0:3] == pytest.approx(obs["body_q"][12:15])


# ---------------------------------------------------------------- 5) カメラ
def test_the_organizer_frame_time_is_used_instead_of_a_local_counter():
    """`obs["t"]` が変わらなければ受信時刻も進まないこと。

    運営 client は新しい frame が無いと `latest()` を返すので `t` が据え置きに
    なる (`components/client.py`)。以前は自前カウンタを入れていたため、
    `camera_stale_roles` の年齢が常に「今」= **止まったカメラを検出できなかった**。
    """
    drv = _driver()

    drv.act(_obs(t=100.0))
    first = drv._head_received_ns
    time.sleep(0.01)
    drv.act(_obs(t=100.0))  # 同じ frame が来続けている
    held = drv._head_received_ns
    time.sleep(0.01)
    drv.act(_obs(t=101.0))  # 新しい frame
    moved = drv._head_received_ns

    assert held == first, "止まっているのに受信時刻が進んでいる"
    assert moved > first, "新しい frame なのに受信時刻が進んでいない"

    # orchestrator に渡る frame にも載っていること。
    assert drv._orch.frames[-1].received_monotonic_ns == moved
    assert drv._orch.frames[0].t != drv._orch.frames[-1].t


def test_a_missing_image_holds_the_previous_frame(capsys):
    """画像が欠けた tick で真っ黒を差し込まないこと。

    真っ黒だと YOLO の検出が 0、policy も真っ黒 wrist で推論を続けるので、
    症状が「動いているのに掴まない」になり切り分けられない。
    """
    drv = _driver()
    img = np.full((480, 640, 3), 200, np.uint8)
    drv.act(
        _obs(t=1.0, images={"ego_view": img, "left_wrist": img, "right_wrist": img})
    )

    drv.act(_obs(t=2.0, images={}))  # 全部欠けた tick

    assert drv._head_bgr is not None
    assert int(drv._head_bgr.max()) == 200, "真っ黒で上書きされている"
    assert drv._wrist_l.get() is not None
    assert int(drv._wrist_l.get().rgb.max()) == 200
    assert "obs に無い" in capsys.readouterr().err


def test_a_head_that_never_arrived_is_reported_as_never_received():
    """1 枚も来ていない状態は受信時刻 0 = `camera_stale_roles` が inf 扱い。"""
    drv = _driver()

    frame = drv._head_frame()

    assert frame.received_monotonic_ns == 0
    assert int(np.asarray(frame.rgb).max()) == 0


# ---------------------------------------------------------------- 1) / 4) HOLD
def test_a_stop_timeout_holds_instead_of_letting_the_expert_keep_running():
    """`on_timeout: stop` で **tick を呼ばなくなる**こと。

    以前は `_advance_halted` が「skill を進めない」だけで、`tick()` は try の外に
    あって回り続けていた。時間切れした expert がそのまま腕を動かし続けるので、
    自前経路の「最後の安全 target を保持して止める」になっていなかった。
    """
    drv = _driver()
    drv.act(_obs(t=1.0))
    assert drv._orch.ticks == 1

    drv._orch.advance_raises = _Safety(
        "skill 'pick_table_leg' reached max_seconds_hard"
    )
    drv.act(_obs(t=2.0))
    assert drv._hold_reason is not None

    drv.act(_obs(t=3.0))
    drv.act(_obs(t=4.0))

    assert drv._orch.ticks == 2, "HOLD 後も推論を続けている"


def test_holding_still_returns_a_valid_taskspace_chunk():
    """HOLD 中も運営へ `(T,25)` を返し続けること (server は落とせない)。"""
    drv = _driver()
    drv._orch.advance_raises = _Safety("停止")
    out = drv.act(_obs())

    actions = np.asarray(out["actions"])
    assert actions.ndim == 2 and actions.shape[1] == 25
    assert np.isfinite(actions).all()


def test_the_held_arms_follow_the_measured_pose():
    """HOLD 中の腕は実測姿勢 (新しい動きを作らない)。"""
    drv = _driver()
    drv._orch.advance_raises = _Safety("停止")
    obs = _obs()
    drv.act(obs)

    drv.act(obs)
    assert drv._orch.ticks == 1
    # 2 回目は tick を呼ばず、実測から組んでいる。
    assert drv._hold_reason == "停止"


def test_an_exception_from_tick_holds_instead_of_killing_the_run(capsys):
    """推論の例外で client 接続が切れないこと。

    `act()` から抜けた例外は `transport.py` に届いて **client が切断される**。
    自前経路は保持して operator を待つので、こちらも保持に倒す。
    """
    drv = _driver()
    drv._orch.tick_raises = RuntimeError("CUDA out of memory")

    out = drv.act(_obs())  # 例外が外へ出ないこと

    assert np.asarray(out["actions"]).shape[1] == 25
    assert drv._hold_reason is not None and "CUDA" in drv._hold_reason
    assert "HOLD" in capsys.readouterr().err


def test_finishing_the_requested_legs_holds():
    """決めた本数を終えたら保持すること (最後の skill を回し続けない)。"""
    drv = _driver()
    drv._orch.state.n_legs_completed = _LEGS

    drv.act(_obs(t=1.0))
    ticks_after_first = drv._orch.ticks
    drv.act(_obs(t=2.0))

    assert drv._hold_reason is not None
    assert drv._orch.ticks == ticks_after_first, "完了後も推論を続けている"


def test_the_hold_is_reported_once(capsys):
    """HOLD の理由は 1 度だけ出す (30Hz で流し続けない)。"""
    drv = _driver()
    drv._orch.tick_raises = RuntimeError("boom")
    for _ in range(5):
        drv.act(_obs())

    assert capsys.readouterr().err.count("[orch-driver] HOLD:") == 1


# ---------------------------------------------------------------- 鮮度の測り方
def test_a_stalled_camera_enters_hold():
    """`obs["t"]` が許容を超えて動かなければ HOLD に入ること。"""
    from components.ramen.orchestrator_driver import _CAMERA_STALE_TIMEOUT_S

    drv = _driver()
    drv.act(_obs(t=100.0))
    assert drv._hold_reason is None

    # 同じ frame が来続ける状態を作る (受信時刻を過去にずらす)。
    drv._frame_received_ns -= int((_CAMERA_STALE_TIMEOUT_S + 0.2) * 1e9)
    drv.act(_obs(t=100.0))

    assert drv._hold_reason is not None
    assert "カメラが止まっている" in drv._hold_reason


def test_a_slow_tick_does_not_look_like_a_stalled_camera():
    """**model のロードで tick が長引いても鮮度で落ちないこと。**

    `tick()` の中では `dispatcher.start()` が model を同期ロードする
    (起動直後の pick は 45 秒)。`_check_camera_freshness` はそのロードの **後** に
    あるので、そこで測ると「45 秒古い frame」に見えて誤発火する
    (2026-09-21 に pod で実測)。判定は取り込んだ瞬間に行う。
    """

    class _SlowOrch(_FakeOrch):
        def tick(self, frame):  # noqa: ANN001
            time.sleep(0.8)  # 許容 0.5s より長い「ロード」
            return super().tick(frame)

    drv = _driver()
    drv._orch = _SlowOrch()

    drv.act(_obs(t=1.0))
    drv.act(_obs(t=2.0))  # 運営は新しい frame を送り続けている

    assert drv._hold_reason is None, drv._hold_reason
    assert drv._orch.ticks == 2


# ---------------------------------------------------------------- head の実ステレオ
#
# 運営 package 2026.09.21 で `ego_view_left` / `ego_view_right` が追加された。
# **pick の expert は HEAD_RIGHT を使う** (`policies/groot_pick_legs.py:70` の 4 cam)
# ので、mono を複製していると学習と違う入力になる。
def test_real_stereo_is_used_when_the_organizer_sends_it():
    """左右が来たら連結してそのまま渡すこと (複製しない)。"""
    drv = _driver()
    left = np.full((480, 640, 3), 10, np.uint8)
    right = np.full((480, 640, 3), 200, np.uint8)
    drv.act(
        _obs(
            t=1.0,
            images={
                "ego_view": np.zeros((480, 640, 3), np.uint8),
                "ego_view_left": left,
                "ego_view_right": right,
                "left_wrist": left,
                "right_wrist": right,
            },
        )
    )

    frame = drv._orch.frames[-1]
    assert frame.rgb.shape == (480, 1280, 3)
    half = frame.rgb.shape[1] // 2
    # BGR に反転して入るので値そのものは 10 / 200 のまま (グレースケール的な塗り)。
    assert int(frame.rgb[:, :half].max()) != int(frame.rgb[:, half:].max()), (
        "左右が同じ = 複製されている"
    )


def test_mono_only_still_falls_back_to_duplication():
    """左右が来ない構成では従来どおり mono を複製すること。"""
    drv = _driver()
    mono = np.full((480, 640, 3), 123, np.uint8)
    drv.act(
        _obs(
            t=1.0,
            images={
                "ego_view": mono,
                "left_wrist": mono,
                "right_wrist": mono,
            },
        )
    )

    frame = drv._orch.frames[-1]
    assert frame.rgb.shape == (480, 1280, 3)
    half = frame.rgb.shape[1] // 2
    assert np.array_equal(frame.rgb[:, :half], frame.rgb[:, half:])


def test_a_half_present_stereo_falls_back_instead_of_guessing():
    """片眼だけ来た tick は複製に落ちること (壊れた連結を作らない)。"""
    drv = _driver()
    mono = np.full((480, 640, 3), 50, np.uint8)
    drv.act(
        _obs(
            t=1.0,
            images={
                "ego_view": mono,
                "ego_view_left": np.full((480, 640, 3), 9, np.uint8),
                "left_wrist": mono,
                "right_wrist": mono,
            },
        )
    )

    frame = drv._orch.frames[-1]
    half = frame.rgb.shape[1] // 2
    assert np.array_equal(frame.rgb[:, :half], frame.rgb[:, half:])


# ---------------------------------------------------------------- 骨盤高さ (col 21)
#
# 運営の adapter は col [21] を WBC へ**リテラル転送**する
# (`wbc_adapter/wbc_driver.py:455` の `base_height_command=[[float(r[21])]]`)。
# 0 は「指定なし」ではなく **骨盤高さ 0 m** = 床まで沈む指令。運営側の既定は
# `wbc_adapter/wbc_goal.py:46` の `DEFAULT_BASE_HEIGHT = 0.74`。
# driver は `groot_chunk_to_taskspace` に base_height を渡さないので、
# ここが実際に会場のワイヤへ出る値そのものになる。
_NEUTRAL_BASE_HEIGHT = 0.74


def test_the_wire_carries_a_neutral_base_height_not_zero():
    """通常 tick の `(T,25)` が col 21 = 0.74 を載せること。"""
    drv = _driver()
    out = drv.act(_obs(t=1.0))

    actions = np.asarray(out["actions"])
    assert actions.shape[1] == 25
    assert actions[:, 21] == pytest.approx(_NEUTRAL_BASE_HEIGHT)


def test_a_held_action_also_carries_the_neutral_base_height():
    """HOLD 中も 0 を出さないこと。

    HOLD は「腕を動かさない」であって「骨盤を床へ」ではない。go-live 直後や
    カメラ停止で最初に出るのが HOLD なので、ここが 0 だと 1 通目から沈む。
    """
    drv = _driver()
    drv._orch.tick_raises = RuntimeError("expert exploded")
    out = drv.act(_obs(t=1.0))

    assert drv._hold_reason is not None
    actions = np.asarray(out["actions"])
    assert actions[:, 21] == pytest.approx(_NEUTRAL_BASE_HEIGHT)


# ---------------------------------------------------------------- Dex1 実測 state
#
# 運営 bridge は 2026-09-21 版から `:5557` に `gripper_q` を載せている
# (`reference/orin_bridge/real_orin_state.py:41-46,67`)。`boundary/states.py` が
# 捨てるので client が 2 本目の SUB で拾い、`obs["gripper_q"]` で渡してくる。
def _gripper(left_q: float, right_q: float) -> dict:
    return {
        "left": {"q": left_q, "dq": 0.0, "tau_est": 0.1},
        "right": {"q": right_q, "dq": 0.0, "tau_est": 0.2},
    }


def test_the_measured_gripper_replaces_the_synthetic_hand_state():
    """実測が来たら合成ではなくそれを model に渡すこと。

    合成は**自分の指令のエコー**なので、滑りも把持失敗も見えない。実測が
    来ているのに使わないと、insert / rotate_leg の state が学習分布から外れる。
    """
    drv = _driver()
    obs = _obs(t=1.0)
    obs["gripper_q"] = _gripper(0.0, -5.30)  # 左が全閉、右が全開
    drv.act(obs)

    state = drv._dex1_src.get()
    assert state.position_rad == pytest.approx([0.0, 4.5])
    assert drv._dex1_src.measured_is_fresh is True


def test_without_a_measured_gripper_the_synthetic_source_still_runs():
    """旧 bridge / 旧 client でも今までどおり動くこと。"""
    drv = _driver()
    drv.act(_obs(t=1.0))  # gripper_q のキー自体が無い

    assert drv._dex1_src.measured_is_fresh is False
    state = drv._dex1_src.get()
    # 合成は開始 skill の frame-0 値から始まる (定数 4.5 ではない)。
    assert state.position_rad == pytest.approx(_INSERT_HAND, abs=1e-3)


def test_the_hybrid_holds_until_the_measured_gripper_arrives():
    """hybrid は実測が来るまで expert を 1 度も走らせないこと。

    区間 1->2 の判定は `実測 - 指令` を見る (interlock.py)。合成は指令のエコーな
    ので差が恒等的に 0 になり `is_grasping` が一度も発火せず、境界が立たないまま
    30 秒で HOLD する。会場では「掴まない」としか見えない。
    """
    drv = _driver()
    drv._hybrid_needs_measured_dex1 = True

    drv.act(_obs(t=1.0))  # 実測なし
    assert drv._orch.ticks == 0
    assert drv._hold_reason is None, "まだ猶予中なので sticky HOLD にはしない"

    obs = _obs(t=2.0)
    obs["gripper_q"] = _gripper(0.0, -5.30)
    drv.act(obs)
    assert drv._orch.ticks == 1


def test_the_hybrid_gives_up_after_the_grace_period():
    """猶予を過ぎても実測が来なければ sticky な HOLD に倒すこと。

    復旧不能な設定ミス (bridge が旧版 / client が載せていない) を、運営スロットの
    中で延々と待ち続けないため。
    """
    from components.ramen.orchestrator_driver import _DEX1_HYBRID_GRACE_TICKS

    drv = _driver()
    drv._hybrid_needs_measured_dex1 = True
    for i in range(_DEX1_HYBRID_GRACE_TICKS + 1):
        drv.act(_obs(t=1.0 + i * 0.05))

    assert drv._orch.ticks == 0
    assert drv._hold_reason is not None
    assert "gripper_q" in drv._hold_reason


def test_the_default_path_is_not_blocked_by_a_missing_gripper():
    """hybrid を使わない既定の GR00T pick は実測が無くても走ること。"""
    drv = _driver()  # _hybrid_needs_measured_dex1 = False
    drv.act(_obs(t=1.0))

    assert drv._orch.ticks == 1
    assert drv._hold_reason is None


def test_the_stereo_keys_can_be_ignored_on_demand():
    """`RAMEN_HEAD_MONO=1` で左右キーを無視して mono 複製に落とせること。

    head カメラが 3840x1080 で開けなかった場合、運営 bridge は警告を出しつつ
    **左右キーを publish し続ける** (`real_orin_cameras.py:166-169`)。その左右は
    「モノラル画像の左半分と右半分」で、中身は違うので運営 preflight の
    byte-identical 検査 (`preflight_sensors.py:163-167`) も通る。
    こちらからは見分けが付かないので、手で落とせる必要がある。
    """
    drv = _driver()
    drv._head_mono = True

    drv.act(
        _obs(
            t=1.0,
            images={
                "ego_view": np.full((480, 640, 3), 50, np.uint8),
                "ego_view_left": np.full((480, 640, 3), 10, np.uint8),
                "ego_view_right": np.full((480, 640, 3), 200, np.uint8),
                "left_wrist": np.zeros((480, 640, 3), np.uint8),
                "right_wrist": np.zeros((480, 640, 3), np.uint8),
            },
        )
    )

    frame = drv._orch.frames[-1]
    half = frame.rgb.shape[1] // 2
    assert np.array_equal(frame.rgb[:, :half], frame.rgb[:, half:]), (
        "RAMEN_HEAD_MONO=1 なのにステレオが使われている"
    )
    assert int(frame.rgb.max()) == 50, "ego_view ではなく左右キーを使っている"


def test_reset_drops_the_measured_gripper_state():
    """episode をまたいで「実測が取れている」を持ち越さないこと。

    warmup の dummy obs (`server._warmup_policy`) は hybrid を空振りさせないために
    `gripper_q` を持つ。それを本番 1 tick 目まで新鮮な実測として残すと、実機で
    実測が来ていなくても hybrid が走り出してしまう。
    """
    drv = _driver()
    obs = _obs(t=1.0)
    obs["gripper_q"] = _gripper(0.0, -5.30)
    drv.act(obs)
    assert drv._dex1_src.measured_is_fresh is True

    drv._orch.reset_episode = lambda: None
    drv._seed_resume_state = lambda: None
    drv.reset()

    assert drv._dex1_src.measured_is_fresh is False
    assert drv._dex1_was_measured is False
    # 「この run で一度は取れた」は診断用に残す。
    assert drv._dex1_src.ever_measured is True


# ---------------------------------------------------------------- (T,25) 生成の失敗
#
# `_taskspace` は以前 try の外にあった。ここで例外が出ると `serve_policy` が
# traceback を client へ返して **接続を切る** (`components/transport.py`)。
# 会場ではそれが run の終わり。「返し続けるが新しい動きは作らない」に倒す。
def test_a_taskspace_failure_holds_instead_of_killing_the_connection(monkeypatch):
    drv = _driver()
    drv.act(_obs(t=1.0))  # 1 本目で正常な行を作っておく
    good = np.asarray(drv._last_actions).copy()

    def _boom(step19, body_q):
        raise ValueError("base_height_cmd 0.0 is outside the plausible range")

    monkeypatch.setattr(drv, "_taskspace", _boom)
    out = drv.act(_obs(t=2.0))

    assert drv._hold_reason is not None and "taskspace" in drv._hold_reason
    assert np.array_equal(np.asarray(out["actions"]), good), "直前の行を返すこと"
    assert np.asarray(out["actions"]).shape[1] == 25


def test_the_hold_path_survives_a_taskspace_failure(monkeypatch):
    """HOLD は最後の砦なので、そこでも例外を外へ出さないこと。"""
    drv = _driver()
    drv.act(_obs(t=1.0))
    good = np.asarray(drv._last_actions).copy()

    drv._enter_hold("test")
    monkeypatch.setattr(
        drv, "_taskspace", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))
    )
    out = drv.act(_obs(t=2.0))

    assert np.array_equal(np.asarray(out["actions"]), good)
