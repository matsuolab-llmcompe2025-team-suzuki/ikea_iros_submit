#!/usr/bin/env python3
"""Exercise interactive controls against a loopback-only, nonphysical follower."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def owned_groot_worker(root_pid, proc_root=Path("/proc")):
    pending = [root_pid]
    matches = []
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        path = proc_root / str(pid)
        try:
            # pixi can spawn Python from a non-main Rust thread. Linux lists
            # children per thread, so inspecting only task/<pid> misses it.
            for children in (path / "task").glob("*/children"):
                pending.extend(int(value) for value in children.read_text().split())
            command = (path / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if (command and b"python" in Path(os.fsdecode(command[0])).name.encode()
                and b"inference.desktop.lower_policy.policies.groot_worker" in command):
            matches.append(pid)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one owned GR00T worker, found {matches}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("retry", "camera", "state", "worker", "next"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    processes, logs, events = [], [], []
    venue = None
    log_path = args.output / "run.log"
    position = 0
    error = None
    wire = None

    def launch(command, log_name, **kwargs):
        log = (args.output / log_name).open("wb")
        logs.append(log)
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        processes.append(proc)
        return proc

    def wait_for(marker, timeout=180):
        nonlocal position
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = log_path.read_text(errors="replace")
            found = text.find(marker, position)
            if found >= 0:
                position = found + len(marker)
                events.append({"marker": marker, "at": time.monotonic()})
                return
            if venue.poll() is not None:
                raise RuntimeError(f"Venue exited {venue.returncode} before {marker!r}")
            time.sleep(0.2)
        raise TimeoutError(f"No {marker!r} within {timeout}s")

    def key(value):
        # The console intentionally rejects keys for 150 ms after a new view
        # and debounces repeats for 350 ms, just like an operator seeing it.
        time.sleep(0.4)
        venue.stdin.write(value)
        venue.stdin.flush()
        events.append({"key": repr(value), "at": time.monotonic()})

    try:
        mock = launch([sys.executable, str(here / "following_mock.py")], "mock.log")
        wire = launch([sys.executable, str(here / "wire_probe.py"),
                       "--output", str(args.output / "wire.json")], "wire.log")
        time.sleep(2)
        if mock.poll() is not None or wire.poll() is not None:
            raise RuntimeError("Mock or observer exited before venue startup")
        stage = "2" if args.case == "next" else "5"
        venue = launch([sys.executable, str(here / "pty_run.py"), "/usr/local/bin/ramen-venue",
                        "--stage", stage, "--actuate"], "run.log", stdin=subprocess.PIPE,
                       env={**os.environ, "IROS_ORIN_HOST": "127.0.0.1"})
        wait_for("Enter starts", 900)
        key(b"\n")
        skill = "rotate_table_base" if args.case == "next" else "flip_table"
        wait_for(skill + "／開始待ち")
        key(b"\n")
        wait_for("フェーズ：テーブル回転" if args.case == "next" else "フェーズ：flip")
        time.sleep(1)
        if args.case == "retry":
            key(b"r")
            wait_for(skill + "／腕保持・ハンド全開")
            key(b"r")
            wait_for(skill + "／開始待ち")
            key(b"\n")
            wait_for("フェーズ：flip")
            time.sleep(2)
            key(b"\x03")
        elif args.case == "next":
            key(b"n")
            wait_for("pick_table_leg／開始待ち")
            key(b"\x03")
        else:
            if args.case == "worker":
                worker_pid = owned_groot_worker(venue.pid)
                os.kill(worker_pid, signal.SIGKILL)
            else:
                fault = signal.SIGUSR1 if args.case == "camera" else signal.SIGUSR2
                mock.send_signal(fault)
            events.append({"fault": args.case, "at": time.monotonic()})
            wait_for("安全停止／保持中・判断待ち", 15)
            time.sleep(2)
            # Restore observation before exercising the explicitly confirmed return.
            if args.case != "worker":
                mock.send_signal(fault)
            time.sleep(1)
            key(b"\n")
            wait_for("operator confirmed the return motion", 10)
        rc = venue.wait(timeout=180)
        expected = (0, 130) if args.case in ("retry", "next") else (1, 2) if args.case == "worker" else (2,)
        if rc not in expected:
            raise RuntimeError(f"Unexpected venue exit {rc}, expected {expected}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # The PTY wrapper forwards TERM and reaps its own child group.
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                error = error or "Process cleanup required SIGKILL"
        for log in logs:
            log.close()
        wire_path = args.output / "wire.json"
        wire_result = json.loads(wire_path.read_text()) if wire_path.exists() else {}
        if not wire_result.get("passed"):
            error = error or "Wire validation failed or absent"
        result = {"case": args.case, "passed": error is None, "error": error,
                  "events": events, "venue_rc": venue.returncode if venue else None,
                  "physical_commands_sent": False, "physics_validated": False}
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    raise SystemExit(0 if error is None else 1)


if __name__ == "__main__":
    main()
