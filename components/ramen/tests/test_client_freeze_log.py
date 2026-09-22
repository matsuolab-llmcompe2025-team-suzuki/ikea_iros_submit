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
  - **bridge が黙った場合も出ない** (それは別の故障で、既存の staleness が扱う)

テストは `time.monotonic` を差し替えて時間を進める。tick 数ではなく経過秒で
判定しているので、呼び出し回数は結果に影響しない。`received_at` は `feed`
fixture が tick ごとに進める = 「bridge から新しい message が届いている」状態。
"""

from __future__ import annotations

import itertools
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


@pytest.fixture
def feed(inference):
    """`_check_frozen` を「bridge から新しい message が届いた tick」として呼ぶ。

    `received_at` を毎回進める。据え置きたい (= bridge が黙った) ときだけ
    明示的に渡す。
    """
    seq = itertools.count(1)

    def _feed(images: dict, received_at: float | None = None) -> None:
        inference._check_frozen(
            images, next(seq) if received_at is None else received_at
        )

    return _feed


def _err(capsys) -> str:
    return capsys.readouterr().err


def test_frozen_camera_warns(feed, clock, capsys):
    """1 秒以上 byte 同一なら警告が出る。"""
    feed({"ego_view": b"frame-A"})
    _err(capsys)

    clock.advance(CAMERA_FREEZE_WARN_S / 2)
    feed({"ego_view": b"frame-A"})
    assert _err(capsys) == "", "閾値未満で鳴ってはいけない"

    clock.advance(CAMERA_FREEZE_WARN_S)
    feed({"ego_view": b"frame-A"})
    err = _err(capsys)
    assert "ego_view" in err
    assert "変わっていない" in err


def test_live_camera_never_warns(feed, clock, capsys):
    """毎 tick 中身が変わっていれば、何秒回しても鳴らない。"""
    for i in range(200):
        clock.advance(0.05)  # 20Hz x 200 = 10 秒
        feed({"ego_view": f"frame-{i}".encode()})
    assert _err(capsys) == ""


def test_rate_mismatch_duplicates_do_not_warn(feed, clock, capsys):
    """publish 30Hz を 20Hz で読む = 同じ JPEG を 2 回続けて見るのが正常。

    ここで鳴ると**健全なカメラで run を落とす**ので、最も守るべき性質。
    """
    frame_no = 0
    for tick in range(200):
        clock.advance(0.05)  # 20Hz で observe
        if tick % 3:  # 3 回に 1 回は新しい frame が来ていない
            frame_no += 1
        feed({"ego_view": f"frame-{frame_no}".encode()})
    assert _err(capsys) == ""


def test_worst_benign_hiccup_does_not_warn(feed, clock, capsys):
    """`cap.read()` の連続失敗による最悪の良性ギャップでも鳴らないこと。

    head thread は読み取りに失敗すると `time.sleep(0.05)` して再試行するだけ
    なので、失敗が続けば capture が数百 ms 止まる。これは故障ではなく、
    **連続一致を N 回で数える設計だとここで誤発火する** (元案は N=3 = 0.15s)。
    閾値直前まで詰めて、鳴らないことを固定する。
    """
    feed({"ego_view": b"hiccup"})
    clock.advance(CAMERA_FREEZE_WARN_S - 0.1)  # 0.9 秒ぶん撮れていない
    for _ in range(18):
        feed({"ego_view": b"hiccup"})
    assert _err(capsys) == ""

    feed({"ego_view": b"recovered"})
    assert _err(capsys) == "", "良性のギャップから復帰しても黙っていること"


def test_warning_repeats_but_is_rate_limited(feed, clock, capsys):
    """凍結が続く間は痕跡を残し続ける。ただし毎 tick は出さない。"""
    feed({"ego_view": b"stuck"})
    clock.advance(CAMERA_FREEZE_WARN_S + 0.1)
    feed({"ego_view": b"stuck"})
    assert "変わっていない" in _err(capsys)

    clock.advance(CAMERA_FREEZE_REPEAT_S / 2)
    feed({"ego_view": b"stuck"})
    assert _err(capsys) == "", "再通知の間隔を守っていない"

    clock.advance(CAMERA_FREEZE_REPEAT_S)
    feed({"ego_view": b"stuck"})
    assert "変わっていない" in _err(capsys)


def test_recovery_reports_how_long_it_was_frozen(feed, clock, capsys):
    """復帰時に継続時間を出す。申告では「いつから いつまで」が要る。"""
    feed({"ego_view": b"stuck"})
    clock.advance(3.0)
    feed({"ego_view": b"stuck"})
    _err(capsys)

    feed({"ego_view": b"moving-again"})
    err = _err(capsys)
    assert "再開" in err
    assert "3.0s" in err


def test_recovery_is_silent_if_it_never_warned(feed, clock, capsys):
    """短い重複から復帰しただけのときは何も出さない。"""
    feed({"ego_view": b"a"})
    clock.advance(CAMERA_FREEZE_WARN_S / 2)
    feed({"ego_view": b"a"})
    feed({"ego_view": b"b"})
    assert _err(capsys) == ""


def test_each_camera_is_tracked_separately(feed, clock, capsys):
    """head だけ凍って wrist は生きている、が実際の故障の形。"""
    for i in range(3):
        clock.advance(CAMERA_FREEZE_WARN_S)
        feed({"ego_view": b"stuck", "left_wrist": f"wrist-{i}".encode()})
    err = _err(capsys)
    assert "ego_view" in err
    assert "left_wrist" not in err


def test_a_silent_bridge_does_not_warn(feed, clock, capsys):
    """bridge が黙った場合は鳴らないこと。**これは別の故障。**

    `observe()` は `read(timeout_ms=0) or latest()` なので、bridge が停止すると
    同じ frame オブジェクトが返り続けて bytes は一致する。しかしそれは
    「カメラの凍結」ではなく「bridge の停止」で、`obs["t"]` が止まるぶん
    **受信時刻ベースの staleness が 0.5 秒で HOLD する**別経路が扱う。

    ここで鳴らすと別の故障を凍結として運営に申告してしまい、申告そのものの
    信用を落とす。
    """
    feed({"ego_view": b"last-frame"})
    for _ in range(50):
        clock.advance(0.1)  # 合計 5 秒、閾値 1.0s を大きく超える
        feed({"ego_view": b"last-frame"}, received_at=7.0)  # 受信時刻が進まない
    assert _err(capsys) == ""


class _FrozenCameras:
    """**受信は継続しているのに中身が同じ** frame を返す `RawCameraStream` 代用。

    これが本物の凍結の形 (bridge の publish loop は capture と非同期なので、
    capture が止まっても message は 30Hz で届き続ける)。
    """

    def __init__(self, jpegs: dict) -> None:
        self._frame = SimpleNamespace(jpegs=jpegs, timestamps={}, received_at=0.0)

    def read(self, timeout_ms: int = 0):
        self._frame.received_at += 1 / 30.0  # 新しい message が届いている
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


def test_missing_key_restarts_the_run(feed, clock, capsys):
    """key ごと消えた期間は凍結に数えない (別経路の話なので)。"""
    feed({"ego_view": b"a"})
    clock.advance(5.0)
    feed({})  # ego_view が obs から落ちた
    feed({"ego_view": b"a"})  # 同じ bytes で再登場
    assert _err(capsys) == "", "消えていた時間を凍結に数えてはいけない"
