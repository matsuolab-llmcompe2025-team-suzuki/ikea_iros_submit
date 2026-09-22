"""`reset()` が例外を外に出さないことを固定する。

`serve_policy` は route の例外を掴むと traceback を client へ返して **接続を切る**
(`components/transport.py` の `except Exception`)。会場ではそれが run の終わり。

`act()` は全例外を HOLD に倒しているのに、`reset()` だけ素通しだった。すると:

    worker が死ぬ -> act() は HOLD で持ちこたえる (設計どおり)
    -> episode 間の reset() -> dispatcher.stop() -> _on_stop() が死んだ worker に
       触る -> 例外 -> 接続断

**安全機構が支えた run を、直後の reset が落とす**形になる (Issue #7)。

reset は「綺麗に始める」ための掃除なので、失敗しても最悪「state が汚いまま次の
episode に入る」で済む。接続を切るより軽い。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_VENDOR = _HERE.parent / "vendor" / "desktop"
for _p in (str(_VENDOR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from components.ramen.orchestrator_driver import OrchestratorDriver  # noqa: E402


class _Boom:
    """呼ばれたら必ず落ちる。"""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        raise self._exc


class _Src:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _Orch:
    def __init__(self, reset_episode) -> None:
        self.reset_episode = reset_episode


def _driver(reset_episode, seed=None) -> OrchestratorDriver:
    """`__init__` を通さずに reset() だけ呼べる最小の driver を組む。

    driver の構築は model の load を伴うので、テストでは属性だけ差し込む。
    """
    d = object.__new__(OrchestratorDriver)
    d._t = 7
    d._advance_halted = True
    d._hold_reason = "前の episode の HOLD"
    d._last_step19 = object()
    d._dex1_src = _Src()
    d._dex1_was_measured = True
    d._orch = _Orch(reset_episode)
    if seed is not None:
        d._seed_resume_state = seed
    else:
        d._seed_resume_state = lambda: None
    return d


def test_reset_clears_the_hold_before_anything_can_fail():
    """掃除の前半 (属性) は、後半が落ちても必ず効くこと。

    `_hold_reason` が残ると次の episode が 1 tick 目から HOLD で始まる。
    """
    d = _driver(_Boom(RuntimeError("worker is gone")))
    d.reset()
    assert d._hold_reason is None
    assert d._advance_halted is False
    assert d._t == 0
    assert d._dex1_src.reset_calls == 1


def test_a_failing_reset_episode_does_not_propagate(capsys):
    """接続を切らないこと。**これが本題。**"""
    boom = _Boom(RuntimeError("dispatcher.stop(): worker is gone"))
    d = _driver(boom)
    d.reset()  # 例外が出れば test は失敗する
    assert boom.calls == 1
    err = capsys.readouterr().err
    assert "orchestrator" in err
    assert "state が汚れたまま" in err


def test_the_rest_still_runs_after_a_failure(capsys):
    """1 つ落ちても残りは進めること (部分的な掃除 > 何もしない)。"""
    seeded = []
    d = _driver(_Boom(RuntimeError("boom")), seed=lambda: seeded.append(True))
    d.reset()
    assert seeded == [True], "reset_episode が落ちたら seed が飛ばされている"


def test_a_failing_seed_also_does_not_propagate(capsys):
    """2 つ目が落ちる場合も同じ。"""
    d = _driver(lambda: None, seed=_Boom(ValueError("RAMEN_START_LEG?")))
    d.reset()
    assert "resume state" in capsys.readouterr().err


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(1)])
def test_even_base_exceptions_are_contained(exc):
    """`BaseException` で受ける。worker の死に方は選べない。"""
    d = _driver(_Boom(exc))
    d.reset()  # 伝播したら失敗
