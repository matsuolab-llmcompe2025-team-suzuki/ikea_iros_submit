"""PC2 read-only state attestation; never publishes a robot command.

The organizer bridge stays untouched on :5557. A separate rt/lowstate
subscriber attests matching bridge samples through a nonce-bound REP endpoint.
Run in the organizer's g1_wbc environment, NOT in the Thor container.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading
import time
import uuid

PROTOCOL = 1
STATE_TOPIC = b"g1_debug"
MAX_SOURCE_AGE_S = 0.5
#: How long one request may wait for this process's own DDS callback to catch up with a
#: bridge payload it has not seen yet. Bounded well inside the Thor reader's 200 ms timeout.
ATTEST_WAIT_S = 0.05
#: A bridge payload is confirmed by a fresh DDS sample of our own that equals it within
#: this bound on every value (joint/gripper rad, quaternion component). The bridge relays
#: one 500 Hz sample; if our callback missed exactly that one, its neighbour 2 ms away
#: still confirms the value. A bridge frozen while the robot moves drifts past the bound
#: within tens of ms; a frozen DDS source leaves no fresh sample at all.
MATCH_TOLERANCE = 0.01
DEFAULT_PROFILE = Path(__file__).with_name("configs") / "pc2_g1_3_internal.json"


class NoFreshMatch(ValueError):
    """No DDS sample seen in the age window matches the bridge payload (yet)."""


def state_values(payload: dict) -> tuple:
    """The 35 compared values: body_q[29], base_quat[4], gripper_q left/right."""
    if not isinstance(payload, dict):
        raise ValueError("state must be a mapping")
    body, quat = payload["body_q"], payload["base_quat"]
    if len(body) != 29 or len(quat) != 4:
        raise ValueError("expected body_q[29] and base_quat[4]")
    values = [*body, *quat, *(payload["gripper_q"][s]["q"] for s in ("left", "right"))]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("state values must be finite numbers")
    return tuple(0.0 if v == 0 else float(v) for v in values)


def state_fingerprint(payload: dict) -> bytes:
    return hashlib.sha256(struct.pack("<35d", *state_values(payload))).digest()


def check_profile(profile: dict, home: Path, env: dict) -> None:
    if profile.get("hand_type") != "internal":
        raise ValueError("this guard supports the internal-hand rig only")
    for relative, expected in profile["files_sha256"].items():
        path = home / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"organizer file missing or changed: {path}; review the rig profile")
    for name, expected in profile["environment"].items():
        actual = env.get(name)
        if actual != expected.replace("${HOME}", str(home)):
            raise ValueError(f"{name} mismatch; source ~/iros_g1_3/iros_env.sh in this shell")
    if env.get("LD_PRELOAD"):
        raise ValueError("LD_PRELOAD belongs to the WBC process only, not the state guard")


class StateAttestor:
    def __init__(self, calibration: dict, *, max_age_s: float = 0.25, clock=time.monotonic):
        if not 0 < max_age_s <= MAX_SOURCE_AGE_S:
            raise ValueError("invalid source age bound")
        for side in ("left", "right"):
            closed, opened = calibration[side]
            if not all(math.isfinite(v) for v in (closed, opened)) or not 4 <= abs(opened-closed) <= 6.5:
                raise ValueError(f"invalid {side} calibration")
        self.calibration, self.max_age_s, self.clock = calibration, max_age_s, clock
        self.session = uuid.uuid4().hex
        self.samples = deque(maxlen=512)
        self.tick = None
        self.sequence = 0
        # Latched only by a tick regression (robot restart). One malformed DDS sample is
        # dropped instead: it never becomes a proof, and it must not stop a healthy source.
        self.fault = None
        self.rejected = None
        self.rejected_count = 0
        self.lock = threading.Lock()

    def observe(self, msg) -> None:
        """SDK callback. A repeated tick cannot renew the source lease."""
        try:
            tick = msg.tick
            if isinstance(tick, bool) or not isinstance(tick, int) or not 0 <= tick < 2**32:
                raise ValueError("missing or invalid DDS tick")
            hands = {}
            for side, index in (("left", 31), ("right", 33)):
                closed, opened = self.calibration[side]
                hands[side] = {"q": (float(msg.motor_state[index].q) - closed) * (-5.30 / (opened-closed))}
            payload = {"body_q": [float(m.q) for m in msg.motor_state[:29]],
                       "base_quat": [float(v) for v in msg.imu_state.quaternion], "gripper_q": hands}
            values = state_values(payload)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError) as exc:
            with self.lock:
                self.rejected = str(exc)
                self.rejected_count += 1
            return
        now = self.clock()
        with self.lock:
            if self.tick is not None:
                delta = (tick - self.tick) % 2**32
                if delta == 0:
                    return
                if delta >= 2**31:
                    self.fault = "DDS tick regressed; inspect robot restart before restarting guard"
                    return
            self.tick = tick
            self.sequence += 1
            self.samples.append((values, now, self.sequence, tick))

    def _why_not(self, reason: str) -> str:
        if self.rejected is None:
            return reason
        return f"{reason}; {self.rejected_count} DDS sample(s) dropped, last: {self.rejected}"

    def attest(self, payload: dict) -> dict:
        """Proof that a fresh DDS sample of our own confirms ``payload``.

        The sample the bridge relayed (exact match) is preferred; otherwise the newest
        sample within ``MATCH_TOLERANCE`` of every value.
        """
        wanted = state_values(payload)
        now = self.clock()
        with self.lock:
            if self.fault:
                raise ValueError(self.fault)
            if self.sequence < 2:
                raise ValueError(self._why_not("waiting for advancing DDS ticks"))
            found = None
            for values, observed, sequence, tick in reversed(self.samples):
                age = now - observed
                if age > self.max_age_s:
                    break
                if age < 0:
                    continue
                if values == wanted:
                    found = (observed, sequence, tick, age, "exact", 0.0)
                    break
                if found is None:
                    diff = max(abs(a - b) for a, b in zip(values, wanted))
                    if diff <= MATCH_TOLERANCE:
                        found = (observed, sequence, tick, age, "tolerance", diff)
            if found is not None:
                observed, sequence, tick, age, match, diff = found
                return {"schema": PROTOCOL, "session": self.session, "sequence": sequence,
                        "tick": tick, "age_s": age, "sample_monotonic_ns": int(observed * 1e9),
                        "match": match, "max_diff": diff}
            reason = self._why_not(
                "bridge state has no fresh matching DDS sample (source stopped or calibration mismatch)"
            )
        raise NoFreshMatch(reason)


def attest_when_seen(guard, payload, *, wait_s=ATTEST_WAIT_S, clock=time.monotonic, sleep=time.sleep):
    """``guard.attest`` that lets this process's DDS callback catch up with the bridge.

    Both processes subscribe to the same ``rt/lowstate``; the bridge may relay a sample a
    few ms before our callback stores it. Waiting briefly turns that race into a match
    instead of a withheld state. A stopped source still fails once the wait is over.
    """
    deadline = clock() + wait_s
    while True:
        try:
            return guard.attest(payload)
        except NoFreshMatch:
            if clock() >= deadline:
                raise
            sleep(0.002)


def serve(guard, profile, bind_address, *, stop_event=None, ready=None):
    """Relay verified state; optional lifecycle hooks are for loopback tests."""
    import msgpack
    import zmq

    context = zmq.Context()
    upstream = context.socket(zmq.SUB)
    endpoint = f"tcp://{bind_address}:{profile['guard_port']}"

    def bind_server():
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, 200)
        for attempt in range(20):  # the closed socket may hold the port for a moment
            try:
                socket.bind(endpoint)
                return socket
            except zmq.ZMQError:
                if attempt == 19:
                    socket.close(linger=0)
                    raise
                time.sleep(0.05)

    server = None
    try:
        upstream.setsockopt(zmq.SUBSCRIBE, STATE_TOPIC)
        upstream.setsockopt(zmq.CONFLATE, 1)
        upstream.setsockopt(zmq.LINGER, 0)
        upstream.connect(profile["upstream_endpoint"])
        server = bind_server()
        if ready is not None:
            ready.set()
        poller = zmq.Poller()
        poller.register(upstream, zmq.POLLIN)
        poller.register(server, zmq.POLLIN)

        def rebuild_server(exc):
            # A REP socket whose request/reply cycle failed cannot be reused.
            # Rebuild it so one failed reply does not end the guard.
            nonlocal server
            print(f"[state-guard] request socket failed ({exc}); rebuilding {endpoint}", flush=True)
            poller.unregister(server)
            server.close(linger=0)
            server = bind_server()
            poller.register(server, zmq.POLLIN)
        payload, received = None, 0.0
        last_error = None
        print("[state-guard] read-only: DDS subscriber + state relay; NO actuator/publisher", flush=True)
        while stop_event is None or not stop_event.is_set():
            events = dict(poller.poll(100))
            if upstream in events:
                try:
                    blob = upstream.recv()
                    payload = msgpack.unpackb(blob[len(STATE_TOPIC):], raw=False)
                    state_fingerprint(payload)
                    received = time.monotonic()
                except (ValueError, TypeError, KeyError, IndexError, struct.error):
                    payload = None
            if server not in events:
                continue
            try:
                request = server.recv_multipart()
            except zmq.ZMQError as exc:
                rebuild_server(exc)
                continue
            nonce = request[0] if request else b""
            try:
                if len(request) != 1 or len(nonce) != 32:
                    raise ValueError("expected a 32-byte request nonce")
                if payload is None or time.monotonic() - received > profile["source_max_age_s"]:
                    raise ValueError("upstream :5557 missing or stale")
                proof = attest_when_seen(guard, payload)
                response = STATE_TOPIC + msgpack.packb(dict(payload, ramen_source=proof), use_bin_type=True)
                last_error = None
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                error = str(exc)
                response = msgpack.packb({"error": error}, use_bin_type=True)
                if error != last_error:
                    print(f"[state-guard] withholding state: {error}", flush=True)
                    last_error = error
            try:
                server.send_multipart([nonce, response])
            except zmq.ZMQError as exc:
                rebuild_server(exc)
    finally:
        upstream.close(linger=0)
        if server is not None:
            server.close(linger=0)
        context.term()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--check-only", action="store_true", help="file/environment audit; no sockets or DDS")
    parser.add_argument("--bind-address", default="127.0.0.1")
    args = parser.parse_args(argv)
    profile = json.loads(args.profile.read_text())
    check_profile(profile, Path.home(), os.environ)
    print(f"[state-guard] profile={profile['name']}; organizer files/environment verified", flush=True)
    if args.check_only:
        return 0

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    guard = StateAttestor(profile["calibration"], max_age_s=profile["source_max_age_s"])
    ChannelFactoryInitialize(0, profile["interface"])
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(guard.observe, 10)
    try:
        serve(guard, profile, args.bind_address)
    except KeyboardInterrupt:
        return 0
    finally:
        subscriber.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
