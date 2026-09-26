"""会場前の確認の道具 (tools/gb10/、VERIFY.md) の test。GPU もネットも使わない。

summarize.py は「外向き接続 0 件」の合否を出す。strace の記録を読み違えて外向きを見落とすと、
会場でネットに出るものを通してしまう (ultralytics の利用統計がそうだった、2026-09-25)。
"""

from __future__ import annotations

import os
import subprocess
import sys
import signal
import time
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


def _actuate_run(tmp_path: Path, log: str, result: str = "rc=0 secs=95 mode=actuate") -> Path:
    run = _run_dir(tmp_path, LOOPBACK_LINES)
    (run / "result.txt").write_text(result + "\n")
    (run / "run.log").write_text(log)
    return run


ACTUATE_LOG = """\
[init] policy variants: flip=groot_flip_table_n17_2_baseline (config)
Harness / E-stop / workspace clearance confirmed. Enter starts Phase 3 stage 5 (...); Ctrl+C cancels:
[go-live] wait_for_go_live_flip_table: waiting for the robot to follow
[go-live] wait_for_go_live_flip_table: still waiting (followed 0.001 / 0.020 rad)
[return] lowering the arms before release
"""


def test_an_actuate_run_that_reached_go_live_and_stopped_on_ctrl_c_passes(tmp_path) -> None:
    result = _summarize(_actuate_run(tmp_path, ACTUATE_LOG))
    assert result.returncode == 0, result.stdout
    assert "[go-live]" in result.stdout


def test_an_error_swallowed_by_the_return_path_fails(tmp_path) -> None:
    """後始末は例外を握るので rc は 0 のまま。log の例外名で落とす (本番 image の np 漏れ)。"""
    log = ACTUATE_LOG + "[return] failed: NameError(\"name 'np' is not defined\"); releasing\n"
    result = _summarize(_actuate_run(tmp_path, log))
    assert result.returncode == 1
    assert "NameError" in result.stdout


#: 故障の後は戻す前に操作者の判断を待ち、run_stage.sh が Enter を送る (本体 #172)
DECISION = """\
Stage 0
フェーズ：安全停止／保持中・判断待ち
操作：Enter 戻す（初期姿勢→ハンド全開→腕下ろし） ｜ Ctrl+C その場で終了
[safety-stop] operator confirmed the return motion
"""

MOCK_STOP_TAIL = """\
[setup] arms are 0.978 rad away from walk_lowered_pose; lowering before the walk
[pre-motion 1/1] waiting: worst=left_elbow target=+0.900 measured=-0.138 error=1.038rad
""" + DECISION + """\
[return] lowering failed: pre-motion stage 'return_forward_outward_clearance' did not converge within 15s
Traceback (most recent call last):
  File "/app/ramen/inference/desktop/entrypoint.py", line 3812, in _run_cli_and_exit
    raise RuntimeError(
RuntimeError: lowering the arms before the walk failed: pre-motion stage 'policy_initial_pose' did not converge within 15s control
"""


def test_the_designed_stop_when_the_mock_does_not_follow_passes(tmp_path) -> None:
    """模擬の PC2 は指令に従わないので Stage 0 は腕を下ろせず、設計どおり RuntimeError で止まる。

    2026-09-25 の GB10 (gb10-test-8c5f4f9) の actuate0 の形。指令の経路はここまでに全部通っている。
    """
    run = _actuate_run(tmp_path, ACTUATE_LOG + MOCK_STOP_TAIL, "rc=1 secs=67 mode=actuate")
    result = _summarize(run)
    assert result.returncode == 0, result.stdout


SAFETY_STOP_TAIL = """\
[safety-stop] operator transition 'arm_pre_motion_for_flip_table' did not reach its target: pre-motion stage 'policy_initial_pose' did not converge within 15s control
[Phase 3 HOLD] Last safe arm and Dex1 targets remain active and walking is stopped.
""" + DECISION + """\
[return] returning to policy frame zero, opening Dex1, then lowering
"""


def test_the_designed_safety_stop_of_stages_1_to_5_passes(tmp_path) -> None:
    """本体 858e107 以降、準備動作が届かないと安全停止 (rc=2)。判断待ちで Enter を受けて戻す。"""
    run = _actuate_run(tmp_path, ACTUATE_LOG + SAFETY_STOP_TAIL, "rc=2 secs=95 mode=actuate")
    result = _summarize(run)
    assert result.returncode == 0, result.stdout


def test_a_designed_stop_without_the_operator_decision_fails(tmp_path) -> None:
    """故障の後に判断を待たずに腕を動かしていたら不合格 (本体 #172 の退行)。"""
    tail = SAFETY_STOP_TAIL.replace(DECISION, "")
    run = _actuate_run(tmp_path, ACTUATE_LOG + tail, "rc=2 secs=95 mode=actuate")
    result = _summarize(run)
    assert result.returncode == 1
    assert "判断を待っていない" in result.stdout


def test_an_unexpected_exception_fails(tmp_path) -> None:
    tail = MOCK_STOP_TAIL.replace(
        "RuntimeError: lowering the arms before the walk failed: pre-motion stage "
        "'policy_initial_pose' did not converge within 15s control",
        "ValueError: boundary action shape mismatch",
    )
    result = _summarize(_actuate_run(tmp_path, ACTUATE_LOG + tail, "rc=1 secs=40 mode=actuate"))
    assert result.returncode == 1
    assert "想定外の例外" in result.stdout


def test_an_actuate_run_that_never_reached_go_live_fails(tmp_path) -> None:
    log = ACTUATE_LOG.replace("[go-live]", "[preflight]")
    result = _summarize(_actuate_run(tmp_path, log))
    assert result.returncode == 1
    assert "go-live 待ちまで進んでいない" in result.stdout


FAKE_ENTRYPOINT = """\
import sys, time
if not sys.stdin.isatty():  # 本物の起動口も --actuate では対話端末を要る (本体 858e107)
    sys.exit("N/R/Enter production controls require an interactive TTY")
print("[init] policy variants: flip=x (config)", file=sys.stderr, flush=True)
input("Harness / E-stop / workspace clearance confirmed. Enter starts Phase 3 stage 5; Ctrl+C cancels: ")
if "--fault" in sys.argv:
    print("[go-live] wait_for_go_live_x: robot followed", file=sys.stderr, flush=True)
    print("[safety-stop] operator transition 'arm_pre_motion_for_x' did not reach its target: "
          "did not converge", file=sys.stderr, flush=True)
    print("フェーズ：安全停止／保持中・判断待ち", file=sys.stderr, flush=True)
    sys.stdin.readline()
    print("[safety-stop] operator confirmed the return motion", file=sys.stderr, flush=True)
    print("[return] returning to policy frame zero", file=sys.stderr, flush=True)
    sys.exit(2)
try:
    while True:
        print("[go-live] wait_for_go_live_x: still waiting", file=sys.stderr, flush=True)
        time.sleep(0.2)
except KeyboardInterrupt:
    print("[return] lowering the arms before release", file=sys.stderr, flush=True)
"""


def test_run_stage_drives_enter_and_ctrl_c_in_actuate_mode(tmp_path) -> None:
    """ACTUATE_HOLD: Enter 1 を送り、go-live 待ちの後に python へ Ctrl+C を送って後始末まで通す。"""
    root = tmp_path / "ramen"
    (root / "inference" / "desktop").mkdir(parents=True)
    (root / "inference" / "__init__.py").write_text("")
    (root / "inference" / "desktop" / "__init__.py").write_text("")
    (root / "inference" / "desktop" / "entrypoint.py").write_text(FAKE_ENTRYPOINT)
    venue = tmp_path / "ramen-venue"
    venue.write_text(f'#!/usr/bin/env bash\nexec {sys.executable} -m inference.desktop.entrypoint "$@"\n')
    venue.chmod(0o755)
    env = {
        **os.environ,
        "RAMEN_ROOT": str(root),
        "RUNS_DIR": str(tmp_path / "runs"),
        "VENUE_BIN": str(venue),
        "NO_MOCK": "1",
        "NOSTRACE": "1",
        "ACTUATE_HOLD": "1",
        "ACTUATE_EXIT_TIMEOUT": "20",
    }
    result = subprocess.run(
        ["bash", str(GB10 / "run_stage.sh"), "5", "stage5_actuate"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    run = tmp_path / "runs" / "stage5_actuate"
    assert result.returncode == 0, result.stderr
    assert (run / "result.txt").read_text().startswith("rc=0 ")
    assert "mode=actuate" in (run / "result.txt").read_text()
    log = (run / "run.log").read_text()
    assert "[go-live]" in log
    assert "[return] lowering" in log, log[-2000:]  # SIGINT が python に届いた
    steps = (run / "steps.log").read_text()
    assert "sending Enter 1" in steps and "SIGTERM" not in steps
    assert _summarize(run).returncode == 0


def test_run_stage_answers_the_safety_stop_decision_on_a_terminal(tmp_path) -> None:
    """安全停止で判断待ちになったら Enter を送る。起動口は擬似端末の上で動く (pty_run.py)。"""
    root = tmp_path / "ramen"
    (root / "inference" / "desktop").mkdir(parents=True)
    (root / "inference" / "__init__.py").write_text("")
    (root / "inference" / "desktop" / "__init__.py").write_text("")
    (root / "inference" / "desktop" / "entrypoint.py").write_text(FAKE_ENTRYPOINT)
    venue = tmp_path / "ramen-venue"
    venue.write_text(f'#!/usr/bin/env bash\nexec {sys.executable} -m inference.desktop.entrypoint "$@"\n')
    venue.chmod(0o755)
    env = {
        **os.environ,
        "RAMEN_ROOT": str(root),
        "RUNS_DIR": str(tmp_path / "runs"),
        "VENUE_BIN": str(venue),
        "NO_MOCK": "1",
        "NOSTRACE": "1",
        "ACTUATE_HOLD": "1",
        "ACTUATE_EXIT_TIMEOUT": "20",
    }
    result = subprocess.run(
        ["bash", str(GB10 / "run_stage.sh"), "5", "stage5_fault", "--fault"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    run = tmp_path / "runs" / "stage5_fault"
    assert result.returncode == 0, result.stderr
    assert (run / "result.txt").read_text().startswith("rc=2 "), (run / "run.log").read_text()
    steps = (run / "steps.log").read_text()
    assert "sending Enter 1" in steps and "sending Enter (return)" in steps
    assert "operator confirmed the return motion" in (run / "run.log").read_text()


def test_the_shell_tools_parse() -> None:
    for script in ("run_stage.sh", "check_envs.sh"):
        result = subprocess.run(
            ["bash", "-n", str(GB10 / script)], capture_output=True, text=True
        )
        assert result.returncode == 0, (script, result.stderr)


def test_pty_preserves_exit_status() -> None:
    result = subprocess.run(
        [sys.executable, str(GB10 / "pty_run.py"), sys.executable, "-c",
         "import sys; assert sys.stdin.isatty(); sys.exit(7)"],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=10,
    )
    assert result.returncode == 7, result.stderr


def test_pty_termination_reaps_its_command(tmp_path) -> None:
    marker = tmp_path / "ready"
    command = (
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, lambda *_: exit(0)); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, str(GB10 / "pty_run.py"), sys.executable, "-c", command],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        child = int(marker.read_text())
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=10)
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("PTY command survived wrapper termination")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_stage_cleanup_does_not_signal_processes_by_name() -> None:
    script = (GB10 / "run_stage.sh").read_text()
    assert "pkill" not in script
    assert "MOCK_PID=$!" in script
