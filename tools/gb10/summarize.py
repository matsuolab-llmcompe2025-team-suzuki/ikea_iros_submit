#!/usr/bin/env python3
"""run_stage.sh の出力を 1 画面にまとめる (VERIFY.md の判定に使う)。

    python3 summarize.py /root/runs/stage1 [/root/runs/stage2 ...]

- result.txt の rc と秒
- 起動口の要所 (VLM の起動・慣らし・vlm_latency、model の読み込み、preflight の合否)
- GPU の使用量の最大 (gpu.log) と MemAvailable の最小 (mem.log)
- 外向き接続: strace の connect のうち loopback / Unix socket 以外。どの process か
  (execve したコマンド) も出す。会場は実行時にネットに出ないので、0 件が合格
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

KEY_LINES = re.compile(
    r"server ready|warm-up done|vlm_latency|integrated GPU|deferred expert ready|"
    r"policy ready|passed; NO command|Error|Traceback"
)
LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "0.0.0.0"}


def _max_column(path: Path) -> float | None:
    values = [
        float(parts[1])
        for parts in (line.split() for line in path.read_text().splitlines())
        if len(parts) == 2
    ]
    return max(values) if values else None


def _min_column(path: Path) -> float | None:
    values = [
        float(parts[1])
        for parts in (line.split() for line in path.read_text().splitlines())
        if len(parts) == 2
    ]
    return min(values) if values else None


def connect_summary(
    path: Path,
) -> tuple[collections.Counter, list[tuple[str, str, str]]]:
    counts: collections.Counter = collections.Counter()
    external: list[tuple[str, str, str]] = []
    commands: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        pid = line.split(None, 1)[0] if line else ""
        if "execve(" in line and line.rstrip().endswith("= 0"):
            argv = re.search(r"\[(.*?)\]", line)
            commands[pid] = (argv.group(1) if argv else line)[:160]
            continue
        if "connect(" not in line:
            continue
        family = re.search(r"sa_family=(AF_\w+)", line)
        family_name = family.group(1) if family else "?"
        if family_name not in ("AF_INET", "AF_INET6"):
            counts[family_name] += 1
            continue
        port = re.search(r"sin6?_port=htons\((\d+)\)", line)
        address = re.search(
            r'inet_addr\("([^"]+)"\)|inet_pton\(AF_INET6, "([^"]+)"', line
        )
        host = (address.group(1) or address.group(2)) if address else "?"
        destination = f"{host}:{port.group(1) if port else '?'}"
        counts[f"{family_name} {destination}"] += 1
        if host not in LOOPBACK:
            external.append(
                (pid, destination, commands.get(pid, "(execve 無し = 親から fork)"))
            )
    return counts, external


def main() -> int:
    failed = False
    for run_dir in map(Path, sys.argv[1:]):
        print(f"== {run_dir.name}: {(run_dir / 'result.txt').read_text().strip()}")
        for line in (run_dir / "run.log").read_text(errors="replace").splitlines():
            if KEY_LINES.search(line) and "help:" not in line:
                print(f"   {line[:170]}")
        gpu = (
            _max_column(run_dir / "gpu.log") if (run_dir / "gpu.log").exists() else None
        )
        mem = (
            _min_column(run_dir / "mem.log") if (run_dir / "mem.log").exists() else None
        )
        if gpu is not None:
            print(f"   GPU 使用量の最大 {gpu / 1024:.1f} GB")
        if mem is not None:
            print(f"   MemAvailable の最小 {mem / 1048576:.1f} GB")
        connect_log = run_dir / "connect.log"
        if connect_log.exists():
            counts, external = connect_summary(connect_log)
            print(
                "   接続先: " + ", ".join(f"{k} x{v}" for k, v in counts.most_common())
            )
            print(f"   外向き connect: {len(external)} 件")
            for pid, destination, command in external[:12]:
                print(f"     pid {pid} -> {destination}  cmd: {command}")
            failed |= bool(external)
        failed |= "rc=0" not in (run_dir / "result.txt").read_text()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
