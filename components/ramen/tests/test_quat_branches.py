"""`rotation_matrix_to_quat_wxyz` の **4 分岐すべて**を検証する。

publish する `(T,25)` の EE 姿勢はここが作る。運営は

    CONTRACT.md:55-58
      "the layout is (w, x, y, z) — but a wrongly-ordered unit quaternion still
       passes this check, so verify your ordering by hand"

と書いていて、**順序も符号も contract 側の検査を素通りする**。

## なぜ 4 分岐を個別に見るか (2026-09-22、Issue #7)

Shepperd 法は trace の符号と対角の大小で 4 通りに分かれる。カバレッジを測ったら
**`trace > 0` の枝しか通っていなかった**。残り 3 本は「大きな回転」で発火する =
腕の姿勢では普通に通る領域。符号ミスが 1 つあると、**特定の姿勢でだけ EE 姿勢が
壊れる**。エラーは出ず、運営 IK が黙って別の場所を解く。

pod の 3012 行との突合 (7.07e-07) は通っているが、あれが 4 分岐を覆っていた保証は
無い (静止シーンの行)。ここで分岐を名指しして固定する。

## 検証方法

quat → 回転行列の**独立な実装**を書いて往復させる。同じ式を使い回すと、
式が間違っていても自己無矛盾に通ってしまう。
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
    rotation_matrix_to_quat_wxyz,
)


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) → 3x3。**上の実装とは独立**に書く (往復が自己無矛盾にならないように)。"""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle(axis, angle: float) -> np.ndarray:
    """Rodrigues。分岐を名指しで踏むために使う。"""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _branch(m: np.ndarray) -> str:
    """実装と同じ判定で、どの枝に入るかを返す。"""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        return "trace"
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        return "xx"
    if m[1, 1] > m[2, 2]:
        return "yy"
    return "zz"


#: 4 分岐を 1 つずつ踏む回転 (軸, 角度)。
_CASES = {
    "trace": ([0.0, 0.0, 1.0], 0.3),
    "xx": ([1.0, 0.0, 0.0], np.pi),
    "yy": ([0.0, 1.0, 0.0], np.pi),
    "zz": ([0.0, 0.0, 1.0], np.pi),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_each_branch_round_trips(name):
    """4 分岐それぞれで、quat から元の回転行列に戻ること。"""
    axis, angle = _CASES[name]
    m = _axis_angle(axis, angle)
    assert _branch(m) == name, f"想定した枝を踏んでいない: {_branch(m)} != {name}"

    q = rotation_matrix_to_quat_wxyz(m)
    assert q.shape == (4,)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-12, "単位 quaternion でない"
    assert q[0] >= 0.0, "w >= 0 の正準形になっていない"

    back = _quat_wxyz_to_matrix(q)
    err = float(np.abs(back - m).max())
    assert err < 1e-9, f"{name} の枝で往復しない (最大 {err:.3e})\n{m}\n{back}"


def test_random_rotations_cover_all_four_branches():
    """乱択でも 4 分岐すべてを踏み、すべて往復すること。

    実機の腕姿勢は全域に散るので、枝を選ばず当たる。
    """
    rng = np.random.default_rng(0)
    seen: dict[str, int] = {}
    worst = 0.0
    for _ in range(2000):
        # 一様ランダム回転 (QR 分解で直交行列を作り、det=+1 に直す)
        q_, r_ = np.linalg.qr(rng.normal(size=(3, 3)))
        q_ = q_ @ np.diag(np.sign(np.diag(r_)))
        if np.linalg.det(q_) < 0:
            q_[:, 0] *= -1.0
        seen[_branch(q_)] = seen.get(_branch(q_), 0) + 1
        back = _quat_wxyz_to_matrix(rotation_matrix_to_quat_wxyz(q_))
        worst = max(worst, float(np.abs(back - q_).max()))
    assert set(seen) == {"trace", "xx", "yy", "zz"}, f"踏めていない枝がある: {seen}"
    assert worst < 1e-9, f"往復誤差が大きい: {worst:.3e}"


def test_w_first_not_w_last():
    """**順序**を固定する。contract の検査は順序違いを通してしまう。

    z 軸まわり 90 度なら (w, x, y, z) = (cos45, 0, 0, sin45)。
    w-last で返していれば先頭が 0 になる。
    """
    q = rotation_matrix_to_quat_wxyz(_axis_angle([0, 0, 1], np.pi / 2))
    expected = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    assert np.allclose(q, expected, atol=1e-12), f"w-first でない: {q}"


def test_identity_is_the_unit_quaternion():
    assert np.allclose(rotation_matrix_to_quat_wxyz(np.eye(3)), [1, 0, 0, 0])


def test_a_non_3x3_input_is_rejected():
    with pytest.raises(ValueError):
        rotation_matrix_to_quat_wxyz(np.eye(4))
