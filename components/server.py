#!/usr/bin/env python3
"""conformance 専用の Thor 側。運営の conformance.py が `components/server.py` として起動する。

**会場の run はこの file を通らない。** docker/venue_entry.sh が本体の entrypoint
(ramen/inference/desktop/entrypoint.py) を直接起動し、その 1 process が PC2 の :5555 / :5557 を
読んで :5556 を bind する (運営 template の「Thor の server + PC2 の client」には分けない。
PC2 の運営 adapter は --actions-host <Thor> で起動する)。

conformance には model を読む時間も Enter も無い (20 s で 40 通) ので、ここでは**本番と同じ
受け口と送り口**だけを使い、指令は「今の実測の姿勢を保つ」にする:

    カメラ :5555  ZmqFrameSource            (最初の 1 枚が届いてから送り始める)
    状態   :5557  BoundaryJointStateSource
    指令   :5556  BoundaryActionSink         (関節角 → FK → (T,25) → 運営の DecoupledSink が検証して送る)

本番が model を読めずに黙ってこの動きに落ちる経路は無い (本番はこの file を import しない)。

    python components/server.py --lane decoupled --port 8765
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

# 本体のコピー (tools/sync_ramen.sh)。repo では submit 直下、image では /app/ramen。
RAMEN_ROOT = Path(__file__).resolve().parents[1] / "ramen"

#: Dex1 の開き [rad] (0 = 閉 / 4.5 = 開)。conformance の mock は手の実測を配らないので開いたまま。
HAND_OPEN_RAD = 4.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", choices=("sonic", "decoupled"), required=True)
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="運営 template の Thor<->Orin の port。私たちの構成では使わない (conformance が渡すので受ける)",
    )
    parser.add_argument(
        "--orin", default="127.0.0.1", help="PC2 (カメラ :5555・状態 :5557) の host"
    )
    parser.add_argument(
        "--action-host", default="127.0.0.1", help=":5556 を bind する host"
    )
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--first-frame-timeout-s", type=float, default=15.0)
    args = parser.parse_args()

    if args.lane != "decoupled":
        print(
            "[server] Team RAMEN は decoupled lane のみ (manifest の lane)",
            file=sys.stderr,
        )
        return 2
    print(
        "[server] conformance 専用: 本番と同じ受け口・送り口で実測の姿勢を保つ指令を送る。"
        "会場の run は docker/venue_entry.sh -> inference.desktop.entrypoint",
        flush=True,
    )

    sys.path.insert(0, str(RAMEN_ROOT))
    import numpy as np

    from inference.desktop.lower_policy.actuators.boundary_sink import (
        BoundaryActionSink,
    )
    from inference.desktop.perception.boundary_state_source import (
        BoundaryJointStateSource,
    )
    from inference.desktop.perception.frame_source import ZmqFrameSource
    from inference.desktop.perception.g1_urdf_fk import G1WristFK

    stopping = False

    def _stop(*_args) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    camera = ZmqFrameSource(
        f"tcp://{args.orin}:5555",
        rcvtimeo_ms=100,
        stereo_view="single",
        image_key="ego_view",
    )
    state = BoundaryJointStateSource(host=args.orin, port=5557)
    sink = BoundaryActionSink(G1WristFK.from_urdf(), port=5556, host=args.action_host)
    sent = 0
    try:
        # 本番と同じく、カメラと状態が届いてから送る
        deadline = time.monotonic() + args.first_frame_timeout_s
        while not stopping and (camera.get() is None or state.get() is None):
            if time.monotonic() > deadline:
                print(
                    f"[server] no camera/state from {args.orin} in {args.first_frame_timeout_s:.0f}s",
                    file=sys.stderr,
                )
                return 1
            time.sleep(0.05)
        print(
            "[server] camera and state are live; holding the measured pose on :5556",
            flush=True,
        )
        period = 1.0 / args.rate_hz
        while not stopping:
            q = np.asarray(state.get().position, dtype=np.float64)
            # 19-D = 腰 3 (実測) + 腕 14 (実測) + 手 2
            action19 = np.concatenate(
                [q[12:15], q[15:29], [HAND_OPEN_RAD, HAND_OPEN_RAD]]
            )
            sink.send_action(action19, q)
            sent += 1
            time.sleep(period)
    finally:
        sink.close()
        state.close()
        camera.close()
        print(f"[server] sent {sent} chunks", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
