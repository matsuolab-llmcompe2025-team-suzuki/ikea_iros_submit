#!/usr/bin/env python3
"""run_stage.sh の出力を 1 画面にまとめる (VERIFY.md の判定に使う)。

    python3 summarize.py /root/runs/stage1 [/root/runs/stage2 ...]

- result.txt の rc と秒
- 起動口の要所 (VLM の起動・慣らし・vlm_latency、model の読み込み、preflight の合否)
- GPU の使用量の最大 (gpu.log) と MemAvailable の最小 (mem.log)
- 外向き接続: strace の connect のうち loopback / Unix socket 以外。どの process か
  (execve したコマンド) も出す。会場は実行時にネットに出ないので、0 件が合格
- --actuate の run (result.txt に mode=actuate、run_stage.sh の ACTUATE_HOLD): Enter 1 の問いと
  go-live 待ち (`[go-live]`) まで進み、log にコードの誤り (NameError 等) が無いのが合格。終わり方は
  Ctrl+C (rc 0 か 130) か、模擬の PC2 が指令に従わないための設計どおりの停止のどちらか:
  - Stage 0: 腕を下ろせないと歩かずに RuntimeError で止まる (rc=1)
  - Stage 1〜5 (本体 858e107 以降): 準備動作が開始姿勢に届かず安全停止 (rc=2、`[safety-stop]
    operator transition … did not reach its target`)
  どちらも本体 #172 以降は、戻す前に「安全停止／保持中・判断待ち」で操作者の判断を待ち、
  run_stage.sh が Enter を送る。判断の行 (`[safety-stop] operator confirmed …` など) が無ければ不合格。
  それ以外の例外は不合格。
  後始末は例外を握って `[return] failed: <例外>` と出すので、rc だけでは誤りが見えない
  (2026-09-25 の本番 image の numpy の import 漏れがそうだった)
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

KEY_LINES = re.compile(
    r"server ready|warm-up done|vlm_latency|integrated GPU|deferred expert ready|"
    r"policy ready|passed; NO command|policy variants|Enter starts|\[go-live\]|\[return\]|"
    r"Error|Traceback"
)
#: 会場のコードの誤り。後始末や判定の中で握られても log には名前が残る
CODE_ERRORS = re.compile(
    r"NameError|AttributeError|TypeError|UnboundLocalError|ImportError|ModuleNotFoundError"
)
#: Traceback の最後の例外の行
EXCEPTION_LINE = re.compile(r"^\w+(?:Error|Exception|Interrupt)\b.*$", re.M)
#: 模擬の PC2 は指令に従わない (関節は勝手に sin 波で動く) ので、腕を動かす準備動作は時間切れになる。
#: 会場ではそこで止めるのが設計 (Stage 0 は腕を下ろせないまま歩かない)
MOCK_STOP = re.compile(r"^RuntimeError: .*did not converge")
#: Stage 1〜5 の設計どおりの停止 (本体 858e107): 準備動作が届かないと次の policy へ進まず安全停止
MOCK_SAFETY_STOP = re.compile(r"\[safety-stop\] operator transition .* did not reach its target")
#: 安全停止・想定外の終了の後に、戻す前に操作者の判断を待った印 (本体 #172)
DECISION_PROMPT = "判断待ち"
DECISION_MADE = re.compile(
    r"\[safety-stop\] (operator confirmed the return motion|operator chose to end|"
    r".*no operator terminal|operator terminal closed)"
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


def actuate_problems(result: str, log: str) -> list[str]:
    """--actuate の run の不合格の理由 (空なら合格)。"""
    problems = []
    final = ""
    if "Traceback (most recent call last):" in log:
        tail = log.rsplit("Traceback (most recent call last):", 1)[1]
        lines = EXCEPTION_LINE.findall(tail)
        final = lines[-1] if lines else "(例外の行が無い)"
    stopped_by_mock = bool(re.search(r"\brc=1\b", result) and MOCK_STOP.match(final))
    safety_stop = bool(
        not final and re.search(r"\brc=2\b", result) and MOCK_SAFETY_STOP.search(log)
    )
    if final and not stopped_by_mock:
        problems.append(f"想定外の例外で止まった: {final[:120]}")
    elif not (stopped_by_mock or safety_stop) and not re.search(r"\brc=(0|130)\b", result):
        problems.append("Ctrl+C でも設計どおりの停止でもない終わり方 (rc)")
    if stopped_by_mock or safety_stop:
        # 故障の後は、戻す前に操作者の判断を待つ (本体 #172)。待たずに腕を動かしていたら不合格
        if DECISION_PROMPT not in log or not DECISION_MADE.search(log):
            problems.append("安全停止の後に操作者の判断を待っていない (判断待ち / 判断の行が無い)")
    if "Enter starts" not in log:
        problems.append("Enter 1 の問いまで進んでいない")
    if "[go-live]" not in log:
        problems.append("go-live 待ちまで進んでいない")
    errors = sorted({m.group(0) for m in CODE_ERRORS.finditer(log)})
    if errors:
        problems.append(f"コードの誤りが log にある: {', '.join(errors)}")
    return problems


def main() -> int:
    failed = False
    for run_dir in map(Path, sys.argv[1:]):
        result = (run_dir / "result.txt").read_text()
        log = (run_dir / "run.log").read_text(errors="replace")
        print(f"== {run_dir.name}: {result.strip()}")
        for line in log.splitlines():
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
        else:
            print("   外向き通信: 未検査 (connect.log がない)")
        if "mode=actuate" in result:
            problems = actuate_problems(result, log)
            for problem in problems:
                print(f"   ✗ {problem}")
            failed |= bool(problems)
        else:
            failed |= not re.search(r"\brc=0\b", result)
            if not re.search(r"\[preflight\].*validation passed; NO command sent", log):
                print("   preflight 完了の記録がない")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
