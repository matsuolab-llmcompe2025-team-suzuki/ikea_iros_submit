"""Thor request/reply reader for the PC2 read-only source-freshness guard."""
from __future__ import annotations

import secrets
import sys
import time

from inference.desktop.perception.venue_state_guard import MAX_SOURCE_AGE_S, PROTOCOL, STATE_TOPIC


class GuardedStateStream:
    def __init__(self, host, port, *, clock_ns=time.monotonic_ns):
        self.endpoint = f"tcp://{host}:{port}"
        self.clock_ns = clock_ns
        self.socket = None
        self.context = None
        self.session = None
        self.sequence = 0
        self.last_sample_ns = None
        self.last_request_ns = 0
        self.failure = None

    def _reset_socket(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None

    def decode_reply(self, nonce, reply, started_ns, received_ns):
        import math
        import msgpack
        from inference.desktop.perception.boundary_state_source import gripper_aware_stream_class
        if len(reply) != 2 or reply[0] != nonce:
            raise ValueError("state guard reply nonce mismatch")
        if not reply[1].startswith(STATE_TOPIC):
            error = msgpack.unpackb(reply[1], raw=False)
            if not isinstance(error, dict):
                raise ValueError("invalid state guard error reply")
            raise ValueError(f"PC2 state guard: {error.get('error', 'invalid reply')}")
        payload = msgpack.unpackb(reply[1][len(STATE_TOPIC):], raw=False)
        if not isinstance(payload, dict):
            raise ValueError("invalid state guard payload")
        proof = payload.get("ramen_source")
        if not isinstance(proof, dict) or proof.get("schema") != PROTOCOL:
            raise ValueError("missing source-freshness proof")
        session, sequence = proof.get("session"), proof.get("sequence")
        tick, sample_ns, age = proof.get("tick"), proof.get("sample_monotonic_ns"), proof.get("age_s")
        if not isinstance(session, str) or len(session) != 32:
            raise ValueError("invalid state guard session")
        if any(type(v) is not int or v < 0 for v in (sequence, tick, sample_ns)) or sequence < 2 or tick >= 2**32:
            raise ValueError("invalid state guard sequence/tick")
        if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or age < 0:
            raise ValueError("invalid source age")
        rtt_s = (received_ns - started_ns) / 1e9
        if rtt_s < 0 or age + rtt_s > MAX_SOURCE_AGE_S:
            raise ValueError("source sample or round-trip is stale")
        if self.session is not None and session != self.session:
            raise ValueError("state guard restarted; restart the Thor run after operator confirmation")
        if sequence <= self.sequence or (self.last_sample_ns is not None and sample_ns <= self.last_sample_ns):
            return None
        state = gripper_aware_stream_class()._decode(None, reply[1])
        if state.gripper_q is None:
            raise ValueError("internal-hand measured state missing; synthetic fallback forbidden")
        # Conservative local-time lower bound; no PC2/Thor wall-clock sync needed.
        state.source_received_ns = started_ns - int(age * 1e9)
        state.source_tick = tick
        self.session, self.sequence, self.last_sample_ns = session, sequence, sample_ns
        return state

    def read(self, timeout_ms=200):
        import zmq
        # Construct/use/close the socket in the receiver thread only.
        if self.context is None:
            self.context = zmq.Context()
        if self.socket is None:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            self.socket.connect(self.endpoint)
        delay = 0.02 - (self.clock_ns() - self.last_request_ns) / 1e9
        if delay > 0:
            time.sleep(delay)
        nonce = secrets.token_bytes(32)
        started = self.clock_ns()
        self.last_request_ns = started
        replied = False
        try:
            self.socket.send(nonce)
            if not self.socket.poll(timeout_ms, zmq.POLLIN):
                raise ValueError("PC2 state guard timed out; confirm the read-only guard on :5558")
            reply = self.socket.recv_multipart()
            replied = True
            state = self.decode_reply(nonce, reply, started, self.clock_ns())
            self.failure = None
            return state
        except (ValueError, TypeError, KeyError, zmq.ZMQError) as exc:
            # A REQ socket stuck without a reply must be rebuilt. After a reply (even a
            # refusal from the guard) the request cycle is complete: reuse the socket
            # rather than reconnecting to PC2 at 50 Hz while the guard withholds state.
            if not replied:
                self._reset_socket()
            reason = str(exc)
            if reason != self.failure:
                print(f"[state-guard] {reason}", file=sys.stderr)
                self.failure = reason
            return None

    def close(self):
        self._reset_socket()
        if self.context is not None:
            self.context.term()
            self.context = None
