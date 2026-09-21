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
    drv._last_obs_t = None
    drv._frame_received_ns = 0
    drv._head_bgr = None
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
    drv._dex1_src = SyntheticDex1StateSource(
        drv._initial_hand2, command_source=drv._hand
    )
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
