#!/usr/bin/env python3
"""Run this before you ship. One command, whole pipeline, pass or fail.

It starts a fake Orin, your server, your client and a fake whole-body
controller on one machine, runs the loop, and checks that what your client
publishes matches the boundary contract.

    python conformance.py --lane sonic
    python conformance.py --lane decoupled

A pass means your wiring is correct: shapes, dtypes, framing, bounds, and
the lane your two halves agreed on. It says nothing about whether your
policy is any good, and it does not exercise real hardware, real latency, or
the e-stop.

Run it against the current template before every bench session — the
contract can move between sessions, and a stale copy fails on the robot
rather than here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable


class Stage:
    """A subprocess that lives for the duration of the check."""

    def __init__(self, name: str, argv: list[str]):
        self.name = name
        self.argv = argv
        # -u throughout: without it a stage's output sits in a pipe buffer and
        # a crash report arrives empty.
        self.process = subprocess.Popen(
            [argv[0], "-u", *argv[1:]], cwd=HERE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    def died(self) -> bool:
        return self.process.poll() is not None

    def drain(self) -> str:
        try:
            out, _ = self.process.communicate(timeout=2)
            return out or ""
        except subprocess.TimeoutExpired:
            self.process.kill()
            out, _ = self.process.communicate()
            return out or ""

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", choices=("sonic", "decoupled"), required=True)
    parser.add_argument("--messages", type=int, default=40,
                        help="Clean action messages required to pass.")
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--verbose", action="store_true",
                        help="Print every stage's output, pass or fail.")
    args = parser.parse_args()

    print(f"conformance: lane={args.lane}, need {args.messages} clean messages\n")

    stages: list[Stage] = []
    verdict = 1
    try:
        # One camera at 10 Hz is plenty to validate the action contract, and
        # keeps four Python processes inside a laptop's memory budget. Use
        # mocks/mock_orin.py directly when you want the full three-camera rig.
        stages.append(Stage("mock-orin", [
            PYTHON, "mocks/mock_orin.py", "--no-wrists", "--fps", "10",
        ]))
        time.sleep(2.0)     # let the PUB sockets bind before anyone subscribes

        stages.append(Stage("server", [
            PYTHON, "components/server.py", "--lane", args.lane, "--port", "8765",
        ]))
        time.sleep(2.0)

        stages.append(Stage("client", [
            PYTHON, "components/client.py", "--lane", args.lane,
            "--thor", "127.0.0.1", "--orin", "127.0.0.1",
        ]))
        time.sleep(1.0)

        checker = subprocess.run(
            [PYTHON, "mocks/mock_wbc.py", "--lane", args.lane,
             "--expect", str(args.messages), "--timeout-s", "20"],
            cwd=HERE, capture_output=True, text=True, timeout=args.timeout_s,
        )
        verdict = checker.returncode

        print(checker.stdout, end="")
        if checker.stderr:
            print(checker.stderr, end="", file=sys.stderr)

        # A stage that died explains a failure better than the checker does.
        for stage in stages:
            if stage.died():
                print(f"\n--- {stage.name} exited early ---")
                print(stage.drain())
                verdict = verdict or 1
            elif args.verbose:
                print(f"\n--- {stage.name} ---")
                print(stage.drain())

    except subprocess.TimeoutExpired:
        print(f"\nTIMEOUT after {args.timeout_s}s — no clean actions reached the "
              f"controller.", file=sys.stderr)
        verdict = 1
    except KeyboardInterrupt:
        verdict = 1
    finally:
        for stage in reversed(stages):
            stage.stop()

    if verdict == 0:
        print("\nPASS — your client speaks the boundary contract.")
        print("Next: confirm your manifest declares lane "
              f"'{args.lane}', then submit.")
    else:
        print("\nFAIL — fix the errors above and run again.", file=sys.stderr)
    return verdict


if __name__ == "__main__":
    sys.exit(main())
