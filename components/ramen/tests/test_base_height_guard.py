"""`(T,25)` の col [21]（骨盤高さ）のガードを固定する。

運営 adapter は col [21] を **WBC へリテラル転送**する
(`reference/wbc_adapter/wbc_driver.py:455`)。つまり `0.0` は「指定なし」ではなく
**「骨盤を高さ 0 m へ」= 床まで沈めろ**という指令。go-live の 1 通目から効く。

実際に `0.0` が数週間どこにも引っかからずに WBC へ届いていた（venuefix で修正）。
運営側でも `boundary/actions.py` でも col [21] は範囲検証されない（hand は [-1,1]、
quat は unit norm を見るのに、下半身列だけ素通し）ので、**止められるのはここだけ**。

既存テストは「中立の 0.74 が出ること」しか見ておらず、**ガード自体を踏んでいなかった**
（2026-09-22 のカバレッジ計測で判明、Issue #7）。ここで固定する。

clamp ではなく raise なのは、黙って直すと「なぜ違う値が出たか」が分からなくなるため。
呼出側（driver）は例外を HOLD に変換するので fail-safe に倒れる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from components.ramen.taskspace_adapter import (  # noqa: E402
    BASE_HEIGHT_MAX_M,
    BASE_HEIGHT_MIN_M,
    DEFAULT_BASE_HEIGHT_M,
    check_base_height,
)


def test_zero_is_rejected():
    """**これが本題。** 0 は「未設定」ではなく「床まで沈め」。"""
    with pytest.raises(ValueError) as e:
        check_base_height(0.0)
    assert "0 means" in str(e.value), "なぜ駄目かがメッセージに無い"


@pytest.mark.parametrize(
    "value", [-1.0, 0.0, 0.29, 1.01, 74.0, float("nan"), float("inf")]
)
def test_out_of_range_is_rejected(value):
    """74 (cm と取り違え) や nan も通さないこと。"""
    with pytest.raises(ValueError):
        check_base_height(value)


@pytest.mark.parametrize(
    "value", [BASE_HEIGHT_MIN_M, DEFAULT_BASE_HEIGHT_M, BASE_HEIGHT_MAX_M]
)
def test_the_plausible_range_passes(value):
    assert check_base_height(value) == pytest.approx(value)


def test_the_default_is_inside_the_range():
    """既定値がガードに弾かれたら、go-live の 1 通目で HOLD する。"""
    assert BASE_HEIGHT_MIN_M < DEFAULT_BASE_HEIGHT_M < BASE_HEIGHT_MAX_M


def test_every_published_row_goes_through_the_guard():
    """**呼び忘れ**を検出する。ガードがあっても呼ばれなければ意味が無い。

    `groot_chunk_to_taskspace` に 0 を渡して、行が出てこないこと
    （= 途中で例外になること）を確かめる。
    """
    from components.ramen.taskspace_adapter import groot_chunk_to_taskspace

    class _Fk:
        def compute_ee_transforms(self, body29):
            eye = np.eye(3)
            return np.zeros(3), eye, np.zeros(3), eye

    chunk = np.zeros((2, 38), dtype=np.float64)
    chunk[:, 3] = 1.0  # root quat を単位に

    # 既定 (0.74) なら通る
    ok = groot_chunk_to_taskspace(chunk, _Fk())
    assert ok.shape == (2, 25)
    assert ok[:, 21] == pytest.approx(DEFAULT_BASE_HEIGHT_M)

    # 0 を明示したら **通らない**
    with pytest.raises(ValueError):
        groot_chunk_to_taskspace(chunk, _Fk(), base_height_cmd=0.0)
