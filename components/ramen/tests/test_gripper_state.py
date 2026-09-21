"""`:5557` の `gripper_q` を読む経路と、model 空間への変換を固定する。

運営の bridge は 2026-09-21 版からグリッパの実測を載せている
(`reference/orin_bridge/real_orin_state.py:41-46,67`) が、`boundary/states.py` の
`_decode` は REQUIRED/OPTIONAL の 4 キーしか取らないので捨てられる。
同じ endpoint に 2 本目の SUB を張って拾う。
"""

from __future__ import annotations

import sys
from pathlib import Path

import msgpack
import numpy as np
import pytest
import zmq

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_VENDOR_DESKTOP = _HERE.parent / "vendor" / "desktop"
for _p in (str(_VENDOR_DESKTOP), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components.ramen.gripper_state import (  # noqa: E402
    STATE_TOPIC,
    GripperStateStream,
)
from components.ramen.orchestrator_io import (  # noqa: E402
    DEX1_OPEN_VALUE,
    DEX1_ORGANIZER_CLOSED_Q,
    DEX1_ORGANIZER_OPEN_Q,
    MeasuredDex1StateSource,
    gripper_q_to_model_rad,
)


def _blob(gripper_q, *, topic: str = STATE_TOPIC) -> bytes:
    """運営 bridge が :5557 に出すのと同じ形の 1 通。"""
    payload = {
        "body_q": [0.0] * 29,
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "gripper_q": gripper_q,
    }
    return topic.encode("utf-8") + msgpack.packb(payload, use_bin_type=True)


def _sides(left_q: float, right_q: float) -> dict:
    return {
        "left": {"q": left_q, "dq": 0.0, "tau_est": 0.1},
        "right": {"q": right_q, "dq": 0.0, "tau_est": 0.2},
    }


# ---------------------------------------------------------------- 単位変換
#
# 往復は `taskspace_adapter.dex1_model_to_taskspace` →
# `run_wbc_with_dex1.hand_norm_to_dex1_q` の逆。両端が「その手の全開/全閉」を
# 指しているので、4.5 と 5.30 という数値の違いはズレではない。
def test_the_endpoints_round_trip_exactly():
    assert gripper_q_to_model_rad(DEX1_ORGANIZER_CLOSED_Q) == pytest.approx(0.0)
    assert gripper_q_to_model_rad(DEX1_ORGANIZER_OPEN_Q) == pytest.approx(
        DEX1_OPEN_VALUE
    )


def test_the_middle_is_the_same_fraction_open():
    """半開は半開に写ること (絶対 rad ではなく開き割合が保たれる)。"""
    half = gripper_q_to_model_rad(DEX1_ORGANIZER_OPEN_Q / 2.0)
    assert half == pytest.approx(DEX1_OPEN_VALUE / 2.0)


def test_out_of_range_motor_angles_are_clamped():
    """較正外の値でも model 空間の外には出さないこと。"""
    assert gripper_q_to_model_rad(+1.0) == pytest.approx(0.0)
    assert gripper_q_to_model_rad(-9.0) == pytest.approx(DEX1_OPEN_VALUE)


def test_the_command_path_and_this_are_inverses():
    """こちらが出した指令が、実測として同じ値で帰ってくること。

    model -> boundary は `dex1_model_to_taskspace`、boundary -> 生モータ角は
    運営の `hand_norm_to_dex1_q` (`tools/run_wbc_with_dex1.py:72-75`)。
    """
    from components.ramen.taskspace_adapter import dex1_model_to_taskspace

    def organizer_hand_norm_to_dex1_q(norm: float) -> float:
        return (
            DEX1_ORGANIZER_CLOSED_Q
            + (DEX1_ORGANIZER_OPEN_Q - DEX1_ORGANIZER_CLOSED_Q) * (1.0 - norm) / 2.0
        )

    for model_rad in (0.0, 1.1, 2.25, 3.9, 4.5):
        norm = dex1_model_to_taskspace(model_rad)
        q = organizer_hand_norm_to_dex1_q(norm)
        assert gripper_q_to_model_rad(q) == pytest.approx(model_rad, abs=1e-6)


# ---------------------------------------------------------------- 受信 (decode)
def test_a_real_bridge_message_is_decoded():
    decoded = GripperStateStream._decode(_blob(_sides(0.0, -5.30)))
    assert decoded == _sides(0.0, -5.30)


def test_a_bridge_without_grippers_decodes_to_none():
    """`motor_state` が 35 未満のリグでは bridge が None を送る。

    `real_orin_state.py:40` の `if n > max(GRIPPER_INDEX.values())`。
    異常ではなく「このリグでは読めない」なので、例外にせず None を返す。
    """
    assert GripperStateStream._decode(_blob(None)) is None


def test_a_partial_entry_is_rejected_instead_of_half_used():
    """片側だけ / キー欠けは捨てること (片手だけ実測にしない)。"""
    assert GripperStateStream._decode(_blob({"left": {"q": 0.0}})) is None
    half = _sides(0.0, -5.30)
    del half["right"]
    assert GripperStateStream._decode(_blob(half)) is None


def test_a_non_finite_angle_is_rejected():
    assert GripperStateStream._decode(_blob(_sides(float("nan"), -5.30))) is None


def test_a_foreign_topic_is_ignored():
    assert GripperStateStream._decode(_blob(_sides(0.0, 0.0), topic="other")) is None


# ---------------------------------------------------------------- state source
def test_the_measured_value_replaces_the_synthetic_one():
    class _Fallback:
        def get(self):
            return "synthetic"

    src = MeasuredDex1StateSource(fallback=_Fallback())
    assert src.get() == "synthetic"
    assert src.ever_measured is False

    assert src.update(_sides(0.0, DEX1_ORGANIZER_OPEN_Q), t=1, obs_t=1.0) is True
    assert src.ever_measured is True
    state = src.get()
    assert np.allclose(state.position_rad, [0.0, DEX1_OPEN_VALUE])


def test_a_missing_gripper_key_falls_back_without_raising():
    """旧 client / 旧 bridge でも run が止まらないこと。"""

    class _Fallback:
        def get(self):
            return "synthetic"

    src = MeasuredDex1StateSource(fallback=_Fallback())
    assert src.update(None, t=1, obs_t=1.0) is False
    assert src.get() == "synthetic"
    assert src.measured_is_fresh is False


def test_a_stale_measurement_falls_back_instead_of_freezing():
    """bridge が止まったら合成に戻ること (古い実測を握り続けない)。"""

    class _Fallback:
        def get(self):
            return "synthetic"

    src = MeasuredDex1StateSource(fallback=_Fallback(), stale_after_s=0.5)
    src.update(_sides(0.0, -5.30), t=1, obs_t=10.0)
    assert src.measured_is_fresh is True

    src.update(None, t=2, obs_t=10.4)  # まだ新鮮
    assert src.measured_is_fresh is True

    src.update(None, t=3, obs_t=10.6)  # 0.5s 超え
    assert src.measured_is_fresh is False
    assert src.get() == "synthetic"


# ---------------------------------------------------------------- 鮮度 (レビュー指摘)
#
# `poll()` が古い値を無期限に返すと、下流の `MeasuredDex1StateSource` は
# 「dict が来た」だけで新鮮さを更新するので **stale_after_s に一度も入れない**。
# `real_orin_state.py` (:5557) だけが落ちて `real_orin_cameras.py` (:5555) が
# 生きている場合、カメラ鮮度でも検出できず、凍結した把持状態を実測として
# 渡し続けることになる。
class _FakeSocket:
    """`recv(NOBLOCK)` だけを差し替えた最小の socket。"""

    def __init__(self, blobs=()):
        self.pending = list(blobs)

    def recv(self, flags=0):
        if not self.pending:
            raise zmq.Again()
        return self.pending.pop(0)

    def close(self, linger=0):
        pass


def _stream(monkeypatch, blobs=(), stale_after_s=0.5):
    s = GripperStateStream.__new__(GripperStateStream)
    s.endpoint = "tcp://test"
    s._socket = _FakeSocket(blobs)
    s._stale_after_s = stale_after_s
    s._latest = None
    s._latest_at = None
    return s


def test_a_frozen_sample_stops_being_returned(monkeypatch):
    """bridge が止まったら poll() が None を返すこと。

    ここで期限切れにしないと `MeasuredDex1StateSource.update()` が毎 tick
    成功扱いになり、`measured_is_fresh` が恒久的に True になる。
    """
    import components.ramen.gripper_state as gs

    now = [100.0]
    monkeypatch.setattr(gs.time, "monotonic", lambda: now[0])

    s = _stream(monkeypatch, [_blob(_sides(0.0, -5.30))], stale_after_s=0.5)
    assert s.poll() is not None, "届いた直後は返る"

    now[0] += 0.4
    assert s.poll() is not None, "0.4s はまだ新鮮"

    now[0] += 0.2  # 合計 0.6s
    assert s.poll() is None, "0.5s を超えたら返さない"
    assert s.age_s == pytest.approx(0.6)


def test_a_fresh_sample_revives_the_stream(monkeypatch):
    """復旧したら また返すこと (一度切れたら終わり、にしない)。"""
    import components.ramen.gripper_state as gs

    now = [100.0]
    monkeypatch.setattr(gs.time, "monotonic", lambda: now[0])

    s = _stream(monkeypatch, [_blob(_sides(0.0, -5.30))], stale_after_s=0.5)
    s.poll()
    now[0] += 1.0
    assert s.poll() is None

    s._socket.pending.append(_blob(_sides(-1.0, -2.0)))
    assert s.poll() is not None
    assert s.age_s == pytest.approx(0.0)


def test_the_downstream_staleness_check_now_fires(monkeypatch):
    """poll() が None になれば measured_is_fresh も落ちること (経路の結合)。"""
    import components.ramen.gripper_state as gs

    now = [100.0]
    monkeypatch.setattr(gs.time, "monotonic", lambda: now[0])

    class _Fallback:
        def get(self):
            return "synthetic"

    stream = _stream(monkeypatch, [_blob(_sides(0.0, -5.30))], stale_after_s=0.5)
    src = MeasuredDex1StateSource(fallback=_Fallback(), stale_after_s=0.5)

    obs_t = 10.0
    src.update(stream.poll(), t=1, obs_t=obs_t)
    assert src.measured_is_fresh is True

    # bridge が止まる。obs (カメラ) は動き続ける
    now[0] += 1.0
    for i in range(20):
        obs_t += 0.05
        src.update(stream.poll(), t=2 + i, obs_t=obs_t)

    assert src.measured_is_fresh is False
    assert src.get() == "synthetic"
