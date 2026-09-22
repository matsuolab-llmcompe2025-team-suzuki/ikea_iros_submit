"""カメラ凍結の検出が **log だけ出して、止めない**ことを固定する。

運営 bridge の publish loop は capture と非同期で、新しい frame が撮れたかを見ずに
30Hz で `_latest_frames` を再エンコードして送る (`real_orin_cameras.py`)。
`timestamps` も publish 時刻なので、capture が止まっても `obs["t"]` は進み、
受信時刻ベースの鮮度チェックは原理的に発火しない。head の capture thread には
再起動が無いので、USB が外れた凍結は回復しない。

一方で **publish 30Hz と capture のレート差による短い重複は正常**に起きる。
だからこの検出は停止条件にしてはならず、閾値も秒で持つ必要がある。ここで
固定するのは次の 2 方向:

  - 本物の凍結 (1 秒以上 byte 同一) では警告が出る
  - 正常なレート差 (短い重複) では **出ない**

テストは `time.monotonic` を差し替えて時間を進める。tick 数ではなく経過秒で
判定しているので、呼び出し回数は結果に影響しない。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_VENDOR_DESKTOP = _HERE.parent / "vendor" / "desktop"
for _p in (str(_VENDOR_DESKTOP), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components.client import (  # noqa: E402
    CAMERA_FREEZE_REPEAT_S,
    CAMERA_FREEZE_WARN_S,
    Inference,
)


class _Clock:
    """差し替え可能な monotonic。"""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(time, "monotonic", c)
    return c


@pytest.fixture
def inference():
    # `_check_frozen` は endpoint を一切触らないので None で足りる。
    return Inference(None, None, None, "", ["ego_view", "left_wrist"])


def _err(capsys) -> str:
    return capsys.readouterr().err


def test_frozen_camera_warns(inference, clock, capsys):
    """1 秒以上 byte 同一なら警告が出る。"""
    inference._check_frozen({"ego_view": b"frame-A"})
    _err(capsys)

    clock.advance(CAMERA_FREEZE_WARN_S / 2)
    inference._check_frozen({"ego_view": b"frame-A"})
    assert _err(capsys) == "", "閾値未満で鳴ってはいけない"

    clock.advance(CAMERA_FREEZE_WARN_S)
    inference._check_frozen({"ego_view": b"frame-A"})
    err = _err(capsys)
    assert "ego_view" in err
    assert "変わっていない" in err


def test_live_camera_never_warns(inference, clock, capsys):
    """毎 tick 中身が変わっていれば、何秒回しても鳴らない。"""
    for i in range(200):
        clock.advance(0.05)  # 20Hz x 200 = 10 秒
        inference._check_frozen({"ego_view": f"frame-{i}".encode()})
    assert _err(capsys) == ""


def test_rate_mismatch_duplicates_do_not_warn(inference, clock, capsys):
    """publish 30Hz を 20Hz で読む = 同じ JPEG を 2 回続けて見るのが正常。

    ここで鳴ると**健全なカメラで run を落とす**ので、最も守るべき性質。
    """
    frame_no = 0
    for tick in range(200):
        clock.advance(0.05)  # 20Hz で observe
        if tick % 3:  # 3 回に 1 回は新しい frame が来ていない
            frame_no += 1
        inference._check_frozen({"ego_view": f"frame-{frame_no}".encode()})
    assert _err(capsys) == ""


def test_worst_benign_hiccup_does_not_warn(inference, clock, capsys):
    """`cap.read()` の連続失敗による最悪の良性ギャップでも鳴らないこと。

    head thread は読み取りに失敗すると `time.sleep(0.05)` して再試行するだけ
    なので、失敗が続けば capture が数百 ms 止まる。これは故障ではなく、
    **連続一致を N 回で数える設計だとここで誤発火する** (元案は N=3 = 0.15s)。
    閾値直前まで詰めて、鳴らないことを固定する。
    """
    inference._check_frozen({"ego_view": b"hiccup"})
    clock.advance(CAMERA_FREEZE_WARN_S - 0.1)  # 0.9 秒ぶん撮れていない
    for _ in range(18):
        inference._check_frozen({"ego_view": b"hiccup"})
    assert _err(capsys) == ""

    inference._check_frozen({"ego_view": b"recovered"})
    assert _err(capsys) == "", "良性のギャップから復帰しても黙っていること"


def test_warning_repeats_but_is_rate_limited(inference, clock, capsys):
    """凍結が続く間は痕跡を残し続ける。ただし毎 tick は出さない。"""
    inference._check_frozen({"ego_view": b"stuck"})
    clock.advance(CAMERA_FREEZE_WARN_S + 0.1)
    inference._check_frozen({"ego_view": b"stuck"})
    assert "変わっていない" in _err(capsys)

    clock.advance(CAMERA_FREEZE_REPEAT_S / 2)
    inference._check_frozen({"ego_view": b"stuck"})
    assert _err(capsys) == "", "再通知の間隔を守っていない"

    clock.advance(CAMERA_FREEZE_REPEAT_S)
    inference._check_frozen({"ego_view": b"stuck"})
    assert "変わっていない" in _err(capsys)


def test_recovery_reports_how_long_it_was_frozen(inference, clock, capsys):
    """復帰時に継続時間を出す。申告では「いつから いつまで」が要る。"""
    inference._check_frozen({"ego_view": b"stuck"})
    clock.advance(3.0)
    inference._check_frozen({"ego_view": b"stuck"})
    _err(capsys)

    inference._check_frozen({"ego_view": b"moving-again"})
    err = _err(capsys)
    assert "再開" in err
    assert "3.0s" in err


def test_recovery_is_silent_if_it_never_warned(inference, clock, capsys):
    """短い重複から復帰しただけのときは何も出さない。"""
    inference._check_frozen({"ego_view": b"a"})
    clock.advance(CAMERA_FREEZE_WARN_S / 2)
    inference._check_frozen({"ego_view": b"a"})
    inference._check_frozen({"ego_view": b"b"})
    assert _err(capsys) == ""


def test_each_camera_is_tracked_separately(inference, clock, capsys):
    """head だけ凍って wrist は生きている、が実際の故障の形。"""
    for i in range(3):
        clock.advance(CAMERA_FREEZE_WARN_S)
        inference._check_frozen(
            {"ego_view": b"stuck", "left_wrist": f"wrist-{i}".encode()}
        )
    err = _err(capsys)
    assert "ego_view" in err
    assert "left_wrist" not in err


class _FrozenCameras:
    """同じ frame を返し続ける `RawCameraStream` 代用。"""

    def __init__(self, jpegs: dict) -> None:
        self._frame = SimpleNamespace(jpegs=jpegs, timestamps={}, received_at=0.0)

    def read(self, timeout_ms: int = 0):
        return self._frame

    def latest(self):
        return self._frame


class _States:
    def read(self, timeout_ms: int = 0):
        return SimpleNamespace(body_q=[0.0] * 29, base_quat=[1.0, 0.0, 0.0, 0.0])

    def latest(self):
        return self.read()


def test_observe_runs_the_check(clock, capsys):
    """**`observe()` から実際に呼ばれている**こと。

    `_check_frozen` を直接叩くテストは「関数が正しい」しか示さない。配線が
    切れていれば本番では 1 行も出ないので、live path から固定する。
    """
    inf = Inference(
        None, _FrozenCameras({"ego_view": b"stuck"}), _States(), "", ["ego_view"]
    )
    assert inf.observe()["images_jpeg"] == {"ego_view": b"stuck"}
    _err(capsys)

    clock.advance(CAMERA_FREEZE_WARN_S + 0.1)
    inf.observe()
    assert "ego_view" in _err(capsys)


def test_missing_key_restarts_the_run(inference, clock, capsys):
    """key ごと消えた期間は凍結に数えない (別経路の話なので)。"""
    inference._check_frozen({"ego_view": b"a"})
    clock.advance(5.0)
    inference._check_frozen({})  # ego_view が obs から落ちた
    inference._check_frozen({"ego_view": b"a"})  # 同じ bytes で再登場
    assert _err(capsys) == "", "消えていた時間を凍結に数えてはいけない"
