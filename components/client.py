#!/usr/bin/env python3
"""運営の conformance.py が起動する `components/client.py`。Team RAMEN の構成では使わない置き物。

運営 template は PC2 (Orin) の client が :5555 / :5557 を読んで Thor に中継し、PC2 で :5556 を
bind する形。私たちは Thor の 1 process がそれを全部やる: PC2 の :5555 / :5557 を直接読み、
:5556 は Thor で bind し、PC2 の運営 adapter を --actions-host <Thor> で起動する。
**PC2 に私たちの物は置かない。** conformance.py がこの path を起動するので、止められるまで待つだけ。

    python components/client.py --lane decoupled --thor 127.0.0.1 --orin 127.0.0.1
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", choices=("sonic", "decoupled"), required=True)
    parser.add_argument("--thor", default="127.0.0.1")
    parser.add_argument("--orin", default="127.0.0.1")
    parser.parse_args()

    print(
        "[client] Team RAMEN では使わない: Thor の process が PC2 の :5555 / :5557 を直接読み、"
        ":5556 を Thor で bind する (components/server.py の docstring)",
        flush=True,
    )
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: stop.set())
    signal.signal(signal.SIGINT, lambda *_args: stop.set())
    stop.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
