#!/usr/bin/env python3
"""コマンドを擬似端末 (pty) の上で動かす (run_stage.sh の --actuate 用)。

会場は `docker run -it` で端末から操作し、起動口は `--actuate` のとき対話端末でないと
「N/R/Enter production controls require an interactive TTY」で起動を拒否する (本体 858e107)。
run_stage.sh は FIFO からキーを送るので、その間に pty を挟む:

    標準入力 (FIFO) → pty → コマンド    コマンドの出力 → pty → 標準出力 (run.log)

終わったら、コマンドの終了コードで終わる。

    python3 pty_run.py <command> [args ...]
"""

from __future__ import annotations

import errno
import os
import pty
import select
import signal
import sys
import time


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: pty_run.py <command> [args ...]", file=sys.stderr)
        return 2
    pid, master = pty.fork()
    if pid == 0:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.execvp(sys.argv[1], sys.argv[1:])

    deadline = None

    def terminate(signum, _frame):
        nonlocal deadline
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5.0

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    inputs = [master, sys.stdin.fileno()]
    status = None
    try:
        while master in inputs:
            if deadline is not None and time.monotonic() >= deadline:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = None
            ready, _, _ = select.select(inputs, [], [], 0.1)
            for fd in ready:
                try:
                    data = os.read(fd, 65536)
                except OSError as exc:
                    if fd != master or exc.errno != errno.EIO:
                        raise
                    data = b""
                if not data:
                    inputs.remove(fd)
                    continue
                destination = sys.stdout.fileno() if fd == master else master
                while data:
                    data = data[os.write(destination, data):]
        _, status = os.waitpid(pid, 0)
    finally:
        os.close(master)
        # Never leave the command running if the wrapper fails or its output is closed.
        if status is None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    # 信号で終わったら shell と同じ 128 + 信号番号 (Ctrl+C = 130)
    return 128 - code if code < 0 else code


if __name__ == "__main__":
    sys.exit(main())
