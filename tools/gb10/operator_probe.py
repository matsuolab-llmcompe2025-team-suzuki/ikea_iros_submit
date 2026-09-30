#!/usr/bin/env python3
"""Exercise interactive controls against a loopback-only, nonphysical follower."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

try:
    from .summarize import CODE_ERRORS, connect_summary
except ImportError:
    from summarize import CODE_ERRORS, connect_summary


def full_stage_plan():
    return {
        1: ["pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"],
        2: ["rotate_table_base", "pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"],
        3: ["rotate_table_base", "pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"],
        4: ["rotate_table_base", "pick_table_leg", "insert_table_leg", "rotate_leg_to_tighten"],
        5: ["flip_table"],
    }


#: 画面の「フェーズ：」に出る skill の名前
PHASE_LABELS = {"rotate_table_base": "テーブル回転", "pick_table_leg": "pick",
                "insert_table_leg": "insert", "rotate_leg_to_tighten": "tighten",
                "flip_table": "flip"}

#: 準備の経由点に遅れて届いたときの操作の行 (段取り表: ready は Enter で準備を続ける)
PREPARATION_READY = "操作：Enter 到達確認・準備を続行"

#: stage を指定して 1 stage の中を流す case
STAGE_CASES = ("stage", "choose", "lift")

# 開始待ちの U / D (本体 #188 skills/start_lift.py) が出す手の高さ。押すたびの 1 行と問いの末尾
_START_LIFT_CM = re.compile(r"\[start-lift\] [^\n+]*\+(\d+(?:\.\d+)?) cm")


def initial_lift_cm(text, skill):
    """起動時の `[init] <skill>: start height keys U/D (…)` から、元の手の高さ [cm]。無ければ None。"""
    match = re.search(
        rf"\[init\] {re.escape(skill)}: start height keys U/D \([^\n+]*\+(\d+(?:\.\d+)?) cm", text)
    return None if match is None else float(match.group(1))


def lift_values(text, start=0):
    """``start`` より後に出た手の高さ [cm] と、その行の終わりの位置。"""
    return [(float(match.group(1)), match.end()) for match in _START_LIFT_CM.finditer(text, start)]


def choice_reply(text, skill, start=0):
    """R の後の数字キーへの返事 (本体 #188 skills/policy_choice.py)。(種類, 行, 終わりの位置) か None。

    種類は accepted (切り替えた・そのまま) / loading (まだ読み込み中) / refused (失敗・解放済み)。
    画面に出る状態の 1 行 (「いま …」) は返事ではないので飛ばす。
    """
    pattern = re.compile(re.escape(f"[model] {skill}: ") + r"[^\r\n]*")
    for match in pattern.finditer(text, start):
        line = match.group(0)
        if "に切り替えた" in line or line.endswith("のまま"):
            return "accepted", line, match.end()
        if "まだ読み込み中" in line:
            return "loading", line, match.end()
        if "選べない" in line:
            return "refused", line, match.end()
    return None


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
    parser.add_argument("--case", choices=("retry", "camera", "gate-camera", "retry-camera", "state", "guard-dds", "worker", "next", "full", "soak", *STAGE_CASES), required=True)
    parser.add_argument("--state-guard", action="store_true",
                        help="Exercise the opt-in :5558 route with the real guard and synthetic DDS")
    parser.add_argument("--skill-config", type=Path,
                        help="Use the exact venue override file instead of the image default")
    parser.add_argument("--stage", type=int, choices=range(6))
    parser.add_argument("--skill", help="choose / lift: the stage's skill to exercise (default: its first)")
    parser.add_argument("--choice-key", default="1", choices=tuple("123456789"),
                        help="choose: the digit pressed after R (a candidate from alternatives_by_skill)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dwell-seconds", type=float, default=30)
    parser.add_argument("--venue-arg", action="append", default=[],
                        help="extra ramen-venue option, repeatable (e.g. --venue-arg=--gpu-models --venue-arg=all)")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.case == "guard-dds" and not args.state_guard:
        parser.error("--case guard-dds requires --state-guard")
    if args.skill_config is not None and not args.skill_config.is_file():
        parser.error("--skill-config must be an existing file")
    if (args.case in STAGE_CASES) != (args.stage is not None):
        parser.error("--stage is required only for --case stage, choose and lift")
    if args.skill is not None and args.case not in ("choose", "lift"):
        parser.error("--skill is only for --case choose and lift")
    if args.case in ("choose", "lift"):
        stage_skills = full_stage_plan().get(args.stage, [])
        args.skill = args.skill or (stage_skills[0] if stage_skills else None)
        if args.skill not in stage_skills:
            parser.error(f"--skill must be one of stage {args.stage}'s skills {stage_skills}")
    if not 1 <= args.dwell_seconds <= 1800:
        parser.error("--dwell-seconds must be in [1,1800]")
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    processes, logs, events = [], [], []
    venue = None
    log_path = args.output / "run.log"
    position = 0
    error = None
    wire = None
    resource_stop = threading.Event()

    def sample_resources():
        with (args.output / "resources.jsonl").open("w") as output:
            while not resource_stop.is_set():
                try:
                    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                    gpu = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10,
                    )
                    sample = {"time": time.time(), "mem_available_kb": int(memory["MemAvailable"].split()[0]),
                              "gpu_processes": gpu.stdout.strip(), "gpu_rc": gpu.returncode}
                except Exception as exc:
                    sample = {"time": time.time(), "error": str(exc)}
                output.write(json.dumps(sample) + "\n")
                output.flush()
                resource_stop.wait(5)

    resource_thread = threading.Thread(target=sample_resources, daemon=True)
    resource_thread.start()

    def launch(command, log_name, **kwargs):
        log = (args.output / log_name).open("wb")
        logs.append(log)
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        processes.append(proc)
        return proc

    # --gpu-models plan (本体 #188) は Enter の後に読みかけの model 1 本を待ってから動き出す
    # (strace 下の VLM の起動で 200 秒ほど)。180 秒では足りないことがある。
    def wait_for(marker, timeout=600, *, retry_enter=False):
        nonlocal position
        deadline = time.monotonic() + timeout
        retry_position = position
        ready_position = position
        retries = 0
        while time.monotonic() < deadline:
            text = log_path.read_text(errors="replace")
            found = text.find(marker, position)
            if found >= 0:
                position = found + len(marker)
                events.append({"marker": marker, "at": time.monotonic()})
                return
            if venue.poll() is not None:
                raise RuntimeError(f"Venue exited {venue.returncode} before {marker!r}")
            # 準備の経由点に遅れて届いた (`準備／<経由点>／ready`) ら、会場の操作者と同じく Enter で続ける
            ready = text.find(PREPARATION_READY, ready_position)
            if ready >= 0:
                ready_position = ready + len(PREPARATION_READY)
                events.append({"preparation_ready_enter": True, "at": time.monotonic()})
                key(b"\n")
            if retry_enter:
                if "安全停止／保持中・判断待ち" in text[position:]:
                    raise RuntimeError("Safety stop while awaiting policy start")
                rejected = text.find("Enter ignored", retry_position)
                if rejected >= 0:
                    retries += 1
                    if retries > 5:
                        raise RuntimeError("Five fresh Enter retries did not satisfy the start gate")
                    retry_position = rejected + len("Enter ignored")
                    events.append({"retry_enter": retries, "reason": "gate rejected previous Enter",
                                   "at": time.monotonic()})
                    key(b"\n")
            time.sleep(0.2)
        raise TimeoutError(f"No {marker!r} within {timeout}s")

    def key(value):
        # The console intentionally rejects keys for 150 ms after a new view
        # and debounces repeats for 350 ms, just like an operator seeing it.
        time.sleep(0.4)
        venue.stdin.write(value)
        venue.stdin.flush()
        events.append({"key": repr(value), "at": time.monotonic()})

    def dwell(stage, skill):
        deadline = time.monotonic() + args.dwell_seconds
        while time.monotonic() < deadline:
            if venue.poll() is not None:
                raise RuntimeError("Venue exited during full-stage dwell")
            if "安全停止／保持中・判断待ち" in log_path.read_text(errors="replace")[position:]:
                raise RuntimeError(f"Safety stop during stage {stage} {skill}")
            time.sleep(0.2)
        events.append({"completed_dwell": skill, "stage": stage,
                       "seconds": args.dwell_seconds, "at": time.monotonic()})

    def wait_lift_change(before, timeout=30):
        nonlocal position
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if venue.poll() is not None:
                raise RuntimeError("Venue exited while changing the start height")
            for cm, end in lift_values(log_path.read_text(errors="replace"), position):
                if cm != before:
                    position = end
                    events.append({"start_lift_cm": cm, "at": time.monotonic()})
                    return cm
            time.sleep(0.2)
        raise TimeoutError(f"Start height did not change from +{before} cm within {timeout}s")

    def exercise_start_lift(skill):
        """開始待ちで U → D (本体 #188)。1 段上がって元に戻るのを見てから Enter へ。"""
        before = initial_lift_cm(log_path.read_text(errors="replace"), skill)
        if before is None:
            raise RuntimeError(f"{skill} offers no U/D start height keys")
        key(b"u")
        raised = wait_lift_change(before)
        key(b"d")
        lowered = wait_lift_change(raised)
        if not raised > before or abs(lowered - before) > 1e-6:
            raise RuntimeError(f"U/D moved the start height +{before} -> +{raised} -> +{lowered} cm")
        time.sleep(3)  # 変えた目標に着いてから Enter (着く前の Enter は無視される)

    def choose_model(skill, choice_key, timeout=900):
        """R の後の画面で数字キー (本体 #188)。読み込み中なら待って押し直し、切り替えたら返す。"""
        nonlocal position
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            key(choice_key.encode())
            reply = None
            reply_deadline = time.monotonic() + 15
            while reply is None and time.monotonic() < reply_deadline:
                if venue.poll() is not None:
                    raise RuntimeError("Venue exited while choosing a model")
                reply = choice_reply(log_path.read_text(errors="replace"), skill, position)
                if reply is None:
                    time.sleep(0.2)
            if reply is None:
                raise TimeoutError(f"No reply to model choice {choice_key!r}")
            kind, line, position = reply
            events.append({"choice": choice_key, "reply": kind, "line": line, "at": time.monotonic()})
            if kind == "accepted":
                if "に切り替えた" not in line:
                    raise RuntimeError(f"Choice {choice_key!r} kept the current model: {line}")
                return
            if kind == "refused":
                raise RuntimeError(f"Model choice refused: {line}")
            time.sleep(10)  # 読み込み中。R の後の保持 (腕が止まっている間) に読み進む
        raise TimeoutError(f"Model choice {choice_key!r} was not accepted within {timeout}s")

    try:
        mock_options = ["--state-guard"] if args.state_guard else []
        mock = launch([sys.executable, str(here / "following_mock.py"), *mock_options], "mock.log")
        wire = launch([sys.executable, str(here / "wire_probe.py"),
                       "--output", str(args.output / "wire.json")], "wire.log")
        deadline = time.monotonic() + 60
        while (time.monotonic() < deadline and mock.poll() is None
               and "mock ready on loopback" not in (args.output / "mock.log").read_text(errors="replace")):
            time.sleep(0.2)
        if mock.poll() is not None or wire.poll() is not None:
            raise RuntimeError("Mock or observer exited before venue startup")
        if "mock ready on loopback" not in (args.output / "mock.log").read_text(errors="replace"):
            raise TimeoutError("Mock did not finish initialization")
        selected_stage = args.stage if args.stage is not None else 2 if args.case == "next" else 5
        stage_args = (["--phase3-full", "--phase3-start-stage", "0", "--phase3-end-stage", "5"]
                      if args.case == "full" else ["--stage", str(selected_stage)])
        # Avoid ptrace stops on unrelated CUDA syscalls while auditing every connect.
        traced = (["strace", "--seccomp-bpf", "-f", "-qq", "-e", "trace=connect,execve", "-o",
                   str(args.output / "connect.log")] if args.trace else [])
        guard_options = ["--boundary-state-guard", "--boundary-state-port", "5558"] if args.state_guard else []
        config_options = ["--skill-config", str(args.skill_config.resolve())] if args.skill_config else []
        venue = launch([sys.executable, str(here / "pty_run.py"), *traced,
                        "/usr/local/bin/ramen-venue", *stage_args, *guard_options, *config_options, "--actuate", *args.venue_arg], "run.log", stdin=subprocess.PIPE,
                       env={**os.environ, "IROS_ORIN_HOST": "127.0.0.1"})
        wait_for("Enter starts", 900)
        key(b"\n")
        if args.case in ("full", *STAGE_CASES):
            if args.case == "full":
                wait_for("Phase 3 continuous stage 0 started")
            plan = full_stage_plan() if args.case == "full" else {
                selected_stage: full_stage_plan().get(selected_stage, [])}
            if args.case in ("choose", "lift"):
                # 確かめる skill で終える (その前の skill は stage と同じく流す)
                skills = plan[selected_stage]
                plan = {selected_stage: skills[:skills.index(args.skill) + 1]}
            for stage, skills in plan.items():
                if args.case == "full":
                    wait_for(f"Phase 3 continuous stage {stage} started", 300)
                for index, skill in enumerate(skills):
                    wait_for(skill + "／開始待ち")
                    if args.case == "lift" and skill == args.skill:
                        exercise_start_lift(skill)
                    key(b"\n")
                    wait_for("フェーズ：" + PHASE_LABELS[skill], retry_enter=True)
                    dwell(stage, skill)
                    if args.case == "choose" and skill == args.skill:
                        # R → 数字キーで候補へ → 初期姿勢 → 開始待ち → 候補の model で policy
                        key(b"r")
                        wait_for(skill + "／腕保持・ハンド全開")
                        choose_model(skill, args.choice_key)
                        wait_for(skill + "／開始待ち")
                        key(b"\n")
                        wait_for("フェーズ：" + PHASE_LABELS[skill], retry_enter=True)
                        dwell(stage, skill)
                    last = (stage == max(plan) and index == len(skills) - 1)
                    key(b"\x03" if last else b"n")
        else:
            skill = "rotate_table_base" if args.case == "next" else "flip_table"
            wait_for(skill + "／開始待ち")
            if args.case != "gate-camera":
                key(b"\n")
                wait_for("フェーズ：テーブル回転" if args.case == "next" else "フェーズ：flip",
                         retry_enter=True)
            if args.case == "retry-camera":
                key(b"r")
                wait_for(skill + "／腕保持・ハンド全開")
            time.sleep(1)
        if args.case == "soak":
            deadline = time.monotonic() + args.dwell_seconds
            while time.monotonic() < deadline:
                if venue.poll() is not None:
                    raise RuntimeError("Venue exited during soak")
                if "安全停止／保持中・判断待ち" in log_path.read_text(errors="replace")[position:]:
                    raise RuntimeError("Safety stop during soak")
                time.sleep(0.2)
            events.append({"completed_dwell": "flip_table", "seconds": args.dwell_seconds,
                           "at": time.monotonic()})
            key(b"\x03")
        elif args.case == "retry":
            key(b"r")
            wait_for(skill + "／腕保持・ハンド全開")
            key(b"r")
            wait_for(skill + "／開始待ち")
            key(b"\n")
            wait_for("フェーズ：flip", retry_enter=True)
            time.sleep(2)
            key(b"\x03")
        elif args.case == "next":
            key(b"n")
            wait_for("pick_table_leg／開始待ち")
            key(b"\x03")
        elif args.case not in ("full", *STAGE_CASES):
            if args.case == "worker":
                worker_pid = owned_groot_worker(venue.pid)
                os.kill(worker_pid, signal.SIGKILL)
            else:
                fault = (signal.SIGWINCH if args.case == "guard-dds" else
                         signal.SIGUSR1 if args.case in ("camera", "gate-camera", "retry-camera") else signal.SIGUSR2)
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
        expected = (0, 130) if args.case in ("retry", "next", "full", "soak", *STAGE_CASES) else (1, 2) if args.case == "worker" else (2,)
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
        resource_stop.set()
        resource_thread.join(timeout=15)
        wire_path = args.output / "wire.json"
        wire_result = json.loads(wire_path.read_text()) if wire_path.exists() else {}
        if not wire_result.get("passed"):
            error = error or "Wire validation failed or absent"
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        if CODE_ERRORS.search(log_text) or "[return] failed:" in log_text or "[return] lowering failed:" in log_text:
            error = error or "Application or return-path error in run.log"
        if args.case == "gate-camera" and "operator confirmed start of flip_table" in log_text:
            error = error or "Obsolete start gate consumed the safety decision Enter"
        network = None
        if args.trace:
            trace_path = args.output / "connect.log"
            counts, external = connect_summary(trace_path) if trace_path.exists() else ({}, [])
            network = {"counts": dict(counts), "external": external}
            if not counts or external:
                error = error or "Missing network observations or outbound connection attempt"
        result = {"case": args.case, "stage": args.stage, "skill": args.skill,
                  "choice_key": args.choice_key if args.case == "choose" else None,
                  "venue_args": args.venue_arg,
                  "state_guard": args.state_guard,
                  "skill_config": str(args.skill_config.resolve()) if args.skill_config else None,
                  "passed": error is None, "error": error,
                  "traced": args.trace, "dwell_seconds": args.dwell_seconds,
                  "network": network,
                  "events": events, "venue_rc": venue.returncode if venue else None,
                  "physical_commands_sent": False, "physics_validated": False}
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    raise SystemExit(0 if error is None else 1)


if __name__ == "__main__":
    main()
