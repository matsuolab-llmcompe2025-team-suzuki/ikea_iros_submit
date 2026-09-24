"""会場前の確認の道具 (tools/gb10/、VERIFY.md) の test。GPU もネットも使わない。

summarize.py は「外向き接続 0 件」の合否を出す。strace の記録を読み違えて外向きを見落とすと、
会場でネットに出るものを通してしまう (ultralytics の利用統計がそうだった、2026-09-25)。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SUBMIT_ROOT = Path(__file__).resolve().parents[1]
GB10 = SUBMIT_ROOT / "tools" / "gb10"

# 2026-09-25 の GB10 の記録から取った形 (VLM・カメラ・Unix socket と、ultralytics の送信)
LOOPBACK_LINES = """\
100 execve("/usr/local/bin/ramen-venue", ["/usr/local/bin/ramen-venue", "--stage", "1"], 0x0) = 0
100 connect(3, {sa_family=AF_INET, sin_port=htons(5555), sin_addr=inet_addr("127.0.0.1")}, 16) = 0
100 connect(4, {sa_family=AF_UNIX, sun_path="/tmp/sock"}, 110) = 0
101 connect(5, {sa_family=AF_INET, sin_port=htons(8000), sin_addr=inet_addr("127.0.0.1")}, 16) = 0
"""
EXTERNAL_LINES = """\
200 execve("/app/ramen/.pixi/envs/runtime/bin/python", ["python", "-m", "inference.desktop.entrypoint"], 0x0) = 0
200 connect(36, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("216.58.205.46")}, 16) = 0
200 connect(36, {sa_family=AF_INET6, sin6_port=htons(443), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2001:4860:4802:36::178", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH
"""


def _run_dir(tmp_path: Path, connect_log: str) -> Path:
    run = tmp_path / "stage1"
    run.mkdir()
    (run / "result.txt").write_text("rc=0 secs=267\n")
    (run / "run.log").write_text(
        "[vlm] server ready after 196.3s\n"
        "[preflight] Phase 3 model/config/4-camera/joint/Dex1 validation passed; NO command sent\n"
    )
    (run / "gpu.log").write_text("1 1024\n2 28979\n3 512\n")
    (run / "mem.log").write_text("1 80000000\n2 50331648\n")
    (run / "connect.log").write_text(connect_log)
    return run


def _summarize(run: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GB10 / "summarize.py"), str(run)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_loopback_and_unix_sockets_pass(tmp_path) -> None:
    result = _summarize(_run_dir(tmp_path, LOOPBACK_LINES))
    assert result.returncode == 0, result.stdout
    assert "外向き connect: 0 件" in result.stdout
    assert "GPU 使用量の最大 28.3 GB" in result.stdout
    assert "MemAvailable の最小 48.0 GB" in result.stdout


def test_an_outbound_connection_fails_and_names_the_process(tmp_path) -> None:
    result = _summarize(_run_dir(tmp_path, LOOPBACK_LINES + EXTERNAL_LINES))
    assert result.returncode == 1
    assert "外向き connect: 2 件" in result.stdout
    assert "216.58.205.46:443" in result.stdout
    assert "2001:4860:4802:36::178:443" in result.stdout
    assert "inference.desktop.entrypoint" in result.stdout


def test_a_failed_run_fails(tmp_path) -> None:
    run = _run_dir(tmp_path, LOOPBACK_LINES)
    (run / "result.txt").write_text("rc=1 secs=442\n")
    assert _summarize(run).returncode == 1


def test_the_shell_tools_parse() -> None:
    for script in ("run_stage.sh", "check_envs.sh"):
        result = subprocess.run(
            ["bash", "-n", str(GB10 / script)], capture_output=True, text=True
        )
        assert result.returncode == 0, (script, result.stderr)
