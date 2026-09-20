"""server.py の 53D re-exec が出力を unbuffered のまま引き継ぐことを確認する。

GPU / model 不要。

## なぜ必要か

image の CMD は `python3 -u components/server.py ...` だが、`os.execv` で
interpreter を張り替えると **-u が落ちる**。stdout が block buffer になり

    [serve] policy server listening on ws://0.0.0.0:8765

が 4KB 溜まるまで出てこない。手順書 (venue_runbook §2-4b) は「この行が出るまで
client を繋ぐな」と指示しており、先に繋ぐと accept rate が near-zero になる。
しかも症状が「Thor を PC2 より先に起動した」場合と同じで切り分けできない。

re-exec するのは 53D を **親プロセス**で読む policy だけ
(`groot_orchestrator` / `groot_53d_real`)。pick(38D) は worker を別 process に
出すので 0.6.0 の親のままでよい。つまり pick だけ試しても踏まない
(2026-09-20 / 21、実 image を載せた RunPod pod で実測)。

## やり方

`server.py` を最後まで import すると numpy / websockets / boundary まで引くので、
module 直下の `_reexec_into_groot53_if_needed()` 呼び出しまでを切り出して exec し、
`os.execv` を差し替えて「どの引数で飛ぼうとしたか」を捕まえる。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SERVER = Path(__file__).resolve().parents[2] / "server.py"
# 行頭の **呼び出し** を狙う。裸の関数名で split すると `def ...()` にも当たる。
_MARKER = "\n_reexec_into_groot53_if_needed()"


def _run_reexec_head(monkeypatch, env: dict[str, str], fake_interp: Path):
    """module 直下の re-exec 判定だけを走らせ、execv の引数を返す (飛ばない)。"""
    src = _SERVER.read_text(encoding="utf-8")
    assert _MARKER in src, "server.py の module 直下の呼び出しが見つからない"
    head = src.split(_MARKER)[0] + _MARKER

    for key in (
        "RAMEN_SERVER_REEXECED",
        "RAMEN_POLICY",
        "RAMEN_WORKER_PYTHON_53D",
        "PYTHONUNBUFFERED",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    captured: dict = {}

    def fake_execv(path, argv):
        captured["path"] = path
        captured["argv"] = list(argv)
        raise _Execed

    monkeypatch.setattr(os, "execv", fake_execv)
    # sys.executable が fake_interp と同じだと「既に 0.6.1」と判断して返るので離す。
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(
        sys, "argv", ["server.py", "--lane", "decoupled", "--port", "8765"]
    )

    try:
        exec(
            compile(head, str(_SERVER), "exec"),
            {"__file__": str(_SERVER), "__name__": "probe"},
        )
    except _Execed:
        pass
    return captured


class _Execed(Exception):
    """execv まで到達した。"""


@pytest.fixture
def fake_interp(tmp_path):
    """0.6.1 venv の python に見せかける実ファイル (is_file() を満たせばよい)。"""
    interp = tmp_path / "venv-groot53" / "bin" / "python"
    interp.parent.mkdir(parents=True)
    interp.write_text("#!/bin/sh\n")
    interp.chmod(0o755)
    return interp


def _orchestrator_env(interp: Path) -> dict[str, str]:
    return {
        "RAMEN_POLICY": "groot_orchestrator",
        "RAMEN_WORKER_PYTHON_53D": str(interp),
    }


def test_reexec_keeps_stdout_unbuffered(monkeypatch, fake_interp):
    """-u を渡し直す。これが無いと listening 行が会場で出ない。"""
    captured = _run_reexec_head(
        monkeypatch, _orchestrator_env(fake_interp), fake_interp
    )

    assert captured, "re-exec が発火していない"
    argv = captured["argv"]
    assert argv[0] == str(fake_interp)
    assert argv[1] == "-u", f"-u が落ちている: {argv}"
    assert argv[2] == str(_SERVER)
    assert argv[3:] == ["--lane", "decoupled", "--port", "8765"], argv


def test_reexec_also_exports_pythonunbuffered(monkeypatch, fake_interp):
    """spawn される worker 側にも効かせる。"""
    _run_reexec_head(monkeypatch, _orchestrator_env(fake_interp), fake_interp)

    assert os.environ.get("PYTHONUNBUFFERED") == "1"


def test_reexec_marks_itself_to_avoid_a_loop(monkeypatch, fake_interp):
    _run_reexec_head(monkeypatch, _orchestrator_env(fake_interp), fake_interp)

    assert os.environ.get("RAMEN_SERVER_REEXECED") == "1"


def test_reexec_also_covers_groot_53d_real(monkeypatch, fake_interp):
    """53D 単体経路も **親プロセス**で 53D を読むので 0.6.1 が要る。

    manifest / venue_runbook が「53D 単体」として案内している経路。ここが
    0.6.0 のままだと flip (Stage 5 の唯一の skill) を含めて全 53D variant が
    `draccus.utils.DecodingError` で落ちる。実 image を載せた pod で
    `FurnitureGrootRuntimeConfig` の DecodingError を実測した (2026-09-21)。
    """
    captured = _run_reexec_head(
        monkeypatch,
        {
            "RAMEN_POLICY": "groot_53d_real",
            "RAMEN_VARIANT": "groot_flip_table_n17_2_baseline",
            "RAMEN_WORKER_PYTHON_53D": str(fake_interp),
        },
        fake_interp,
    )

    assert captured, "groot_53d_real で re-exec が発火していない"
    assert captured["argv"][:2] == [str(fake_interp), "-u"], captured["argv"]


def test_no_reexec_for_the_pick_policy(monkeypatch, fake_interp):
    """pick(38D) は 0.6.0 の親で動くので張り替えない。"""
    captured = _run_reexec_head(
        monkeypatch,
        {
            "RAMEN_POLICY": "groot_pick_real",
            "RAMEN_WORKER_PYTHON_53D": str(fake_interp),
        },
        fake_interp,
    )

    assert captured == {}, f"pick で re-exec してしまっている: {captured}"
