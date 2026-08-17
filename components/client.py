#!/usr/bin/env python3
"""Your policy client, running on the Jetson Orin NX onboard the G1.

    Orin 192.168.100.2   this file + the organizer's three endpoints
    Thor 192.168.100.1   components/server.py

Adapt this freely — it is yours. It exists to show the shape of the loop:

    boundary.CameraStream  :5555 ─┐
                                  ├─> observation ─> Thor ─> action chunk ─┐
    boundary.StateStream   :5557 ─┘                                        │
                                                boundary.ActionSink :5556 <┘

    # on the Orin
    python components/client.py --lane sonic --thor 192.168.100.1

--------------------------------------------------------------------------
Why this loop is pipelined
--------------------------------------------------------------------------
The naive loop — query, execute the whole chunk, query again — leaves the
robot executing stale actions for one full inference latency at the seam of
every chunk. At ~200 ms of round trip that is ten dead control periods, and
it shows up as a visible stutter between chunks.

So this client does two things instead:

  1. PREFETCH. The next inference is submitted on a worker thread while the
     current chunk is still playing out, so a fresh chunk is usually already
     in hand when the old one runs dry.

  2. LATENCY COMPENSATION. A chunk that took L seconds to compute describes
     the world as it was L seconds ago, so its first round(L * rate) rows
     have already been overtaken by events. They are skipped rather than
     replayed.

Both matter more the slower your model is. Tune with --prefetch-rows.

--------------------------------------------------------------------------
Safety
--------------------------------------------------------------------------
Killing or pausing this client does NOT stop the robot. The whole-body
controller keeps replaying the last command it got. The only thing that
brings the robot to a safe state is the organizer's independent e-stop,
which damps the motors directly and does not go through your code. Never
treat "my client exited" as "the robot stopped".

NETWORK: the Thor<->Orin link is a direct ethernet cable with static IPs. If
the client cannot reach the Thor, check `ip link` for the interface being
down before you go looking at code.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boundary import ActionSink, CameraStream, StateStream  # noqa: E402
from boundary.actions import ActionError  # noqa: E402
from components.transport import PolicyLink  # noqa: E402

LANES = ("sonic", "decoupled")
SONIC_STEP_HZ = 50.0        # gear_sonic_deploy's control cadence
DECOUPLED_CHUNK_HZ = 20.0   # task-space re-query rate


class Inference:
    """Runs one inference at a time on a worker thread, so the control loop
    never blocks on the network.

    ``submit`` snapshots the observation on the calling thread — sampling it
    inside the worker would time-stamp the observation at whenever the thread
    happened to start, which is exactly the error latency compensation is
    trying to correct.
    """

    def __init__(self, link: PolicyLink, cameras, states, prompt, camera_keys):
        self._link = link
        self._cameras = cameras
        self._states = states
        self._prompt = prompt
        self._camera_keys = camera_keys
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._pending: Future | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def observe(self) -> dict | None:
        """Sample both input endpoints. None until each has produced once."""
        frame = self._cameras.read(timeout_ms=0) or self._cameras.latest()
        state = self._states.read(timeout_ms=0) or self._states.latest()
        if frame is None or state is None:
            return None
        # Only the cameras the server declared: three 480x640x3 images is
        # ~2.7 MB per step raw, so shipping ones the model ignores is pure
        # latency.
        return {
            "images": {k: frame.images[k] for k in self._camera_keys if k in frame.images},
            "body_q": state.body_q,
            "base_quat": state.base_quat,
            "prompt": self._prompt,
            "t": frame.received_at,
        }

    def submit(self) -> bool:
        """Kick off the next inference. False if one is already in flight or
        the endpoints have not warmed up yet."""
        if self._pending is not None:
            return False
        obs = self.observe()
        if obs is None:
            return False
        issued = time.monotonic()
        self._pending = self._pool.submit(
            lambda: (self._link.act(obs), time.monotonic() - issued)
        )
        return True

    def ready(self) -> bool:
        return self._pending is not None and self._pending.done()

    def collect(self, block: bool = True) -> tuple[dict, float] | None:
        """Take the finished chunk and its measured latency in seconds."""
        if self._pending is None:
            return None
        if not block and not self._pending.done():
            return None
        try:
            return self._pending.result()
        finally:
            self._pending = None

    def close(self):
        # cancel_futures は Python 3.9+ で追加。Orin (JetPack 5.1.1) の default Python は
        # 3.8 で未対応 → shutdown が TypeError で exit path を crash させる (運営 onboarding
        # Finding 3、upstream template の bug)。version guard で回避。
        if sys.version_info >= (3, 9):
            self._pool.shutdown(wait=False, cancel_futures=True)
        else:
            self._pool.shutdown(wait=False)


_warned_short_chunk = False


def _skip_rows(latency_s: float, rate_hz: float, chunk_length: int) -> int:
    """How many leading rows of a fresh chunk are already stale.

    Always leaves at least one row: a chunk that took longer to compute than
    it lasts is a sign your model is too slow for the cadence, not a reason
    to send nothing.
    """
    global _warned_short_chunk
    skip = min(int(round(latency_s * rate_hz)), max(chunk_length - 1, 0))
    if not _warned_short_chunk and skip * 2 > chunk_length:
        _warned_short_chunk = True
        print(
            f"[client] WARNING: {skip} of {chunk_length} rows are stale on arrival "
            f"({latency_s * 1000:.0f}ms at {rate_hz:.0f}Hz). Most of every chunk is "
            f"being discarded. Return a longer chunk (>= {int(latency_s * rate_hz * 3)} "
            f"rows) or make inference faster, or the robot will spend most of its "
            f"time replaying the tail of an old chunk.",
            file=sys.stderr,
        )
    return skip


def run_sonic(inference: Inference, sink, prefetch_rows: int):
    """Stream latent rows at 50 Hz, prefetching the next chunk mid-playback."""
    period = 1.0 / SONIC_STEP_HZ
    token = left = right = None
    index = 0
    chunks = 0

    while True:
        # Out of rows: block for the chunk that should already be in flight.
        if token is None or index >= len(token):
            if not inference.busy:
                while not inference.submit():
                    print("[client] waiting for camera/state...")
                    time.sleep(0.1)
            action, latency = inference.collect(block=True)
            token = action["motion_token"]
            left = action["left_hand_joints"]
            right = action["right_hand_joints"]
            # Validate the whole chunk before any of it reaches the robot, so
            # a bad tail cannot be half-executed.
            token, left, right = sink.validate_chunk(token, left, right)
            index = _skip_rows(latency, SONIC_STEP_HZ, len(token))
            chunks += 1
            if chunks % 10 == 1:
                print(f"[client] chunk T={len(token)} latency={latency * 1000:.0f}ms "
                      f"skip={index} peak|token|={float(abs(token).max()):.3f}")

        # Enough rows left to cover the round trip? Start the next inference.
        if not inference.busy and (len(token) - index) <= prefetch_rows:
            inference.submit()

        tick = time.monotonic()
        sink.send_step(token[index:index + 1], left[index:index + 1], right[index:index + 1])
        index += 1
        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)


def run_decoupled(inference: Inference, sink):
    """Publish a task-space chunk per tick, with the next one already in flight."""
    period = 1.0 / DECOUPLED_CHUNK_HZ
    chunks = 0

    while not inference.submit():
        print("[client] waiting for camera/state...")
        time.sleep(0.1)

    while True:
        tick = time.monotonic()
        action, latency = inference.collect(block=True)
        chunk = sink.validate_chunk(action["actions"])

        # The adapter owns interpolation, so hand it the trajectory minus the
        # rows that inference latency already consumed.
        skip = _skip_rows(latency, DECOUPLED_CHUNK_HZ, len(chunk))
        sink.send_chunk(chunk[skip:], issued_at=time.time() - latency)

        # Submit the next one before sleeping, not after.
        inference.submit()

        chunks += 1
        if chunks % 10 == 1:
            print(f"[client] chunk T={len(chunk)} latency={latency * 1000:.0f}ms skip={skip}")

        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", choices=LANES,
                        default=os.environ.get("PEVAL_LANE", "sonic"))
    parser.add_argument("--thor", default="192.168.100.1", help="Policy server host.")
    parser.add_argument("--thor-port", type=int, default=8765)
    parser.add_argument("--orin", default="127.0.0.1",
                        help="Host of the organizer's camera/state endpoints.")
    parser.add_argument("--prompt", default="grab the bottle",
                        help="Task instruction handed to the policy.")
    parser.add_argument("--prefetch-rows", type=int, default=8,
                        help="SONIC only: rows left in the current chunk when the "
                             "next inference starts. Raise it if your model is slow.")
    args = parser.parse_args()

    cameras = CameraStream(host=args.orin)
    states = StateStream(host=args.orin)
    sink = ActionSink.for_lane(args.lane)

    print(f"[client] lane={args.lane} thor={args.thor}:{args.thor_port} orin={args.orin}")
    print("[client] waiting for the organizer's endpoints...")
    cameras.wait_until_live()
    state = states.wait_until_live()
    print(f"[client] endpoints live (hand state "
          f"{'present' if state.hands_present else 'absent — Dex1-1 rig'})")

    link = PolicyLink(f"ws://{args.thor}:{args.thor_port}")
    declared = link.metadata.get("lane")
    if declared != args.lane:
        raise SystemExit(
            f"[client] lane mismatch: server declared {declared!r}, client is "
            f"{args.lane!r}. Start both on the same lane."
        )

    link.reset()
    inference = Inference(link, cameras, states, args.prompt,
                          link.metadata.get("camera_keys", ["ego_view"]))
    try:
        if args.lane == "sonic":
            run_sonic(inference, sink, args.prefetch_rows)
        else:
            run_decoupled(inference, sink)
    except KeyboardInterrupt:
        print("\n[client] interrupted — THE ROBOT IS STILL HOLDING ITS LAST COMMAND. "
              "Use the e-stop to bring it to a safe state.")
    except ActionError as exc:
        print(f"\n[client] ACTION REJECTED: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        inference.close()
        sink.close()
        cameras.close()
        states.close()


if __name__ == "__main__":
    main()
