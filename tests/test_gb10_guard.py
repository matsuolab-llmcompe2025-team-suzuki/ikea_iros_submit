"""Loopback integration for the exact opt-in state path; no robot or GPU."""

from pathlib import Path
import socket
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ramen"))
sys.path.insert(0, str(ROOT / "tools" / "gb10"))

from guard_mock import LoopbackStateGuard
from inference.desktop.perception.guarded_state_stream import GuardedStateStream


@pytest.fixture
def rig():
    import msgpack
    import zmq
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    upstream_port = publisher.bind_to_random_port("tcp://127.0.0.1")
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    guard = LoopbackStateGuard(upstream_port=upstream_port, guard_port=port)
    reader = GuardedStateStream("127.0.0.1", port)
    payload = {"body_q": [.2] * 29, "base_quat": [1., 0., 0., 0.],
               "gripper_q": {"left": {"q": -2.65}, "right": {"q": -4.5}}}

    def exchange(*, dds=True, bridge=True):
        if dds:
            guard.observe(payload)
        if bridge:
            publisher.send(b"g1_debug" + msgpack.packb(payload))
        return reader.read(timeout_ms=200)

    def fresh():
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = exchange()
            if state is not None:
                return state
        pytest.fail(f"No fresh guarded state: {reader.failure}")

    try:
        yield guard, reader, exchange, fresh, payload
    finally:
        reader.close()
        guard.close()
        publisher.close()
        context.term()


def test_real_guard_preserves_canonical_internal_hand_state(rig):
    _, _, _, fresh, payload = rig
    first = fresh()
    second = fresh()
    assert first.body_q.tolist() == pytest.approx(payload["body_q"])
    assert first.gripper_q.tolist() == pytest.approx([-2.65, -4.5])
    assert second.source_tick > first.source_tick
    assert second.source_sample_ns > first.source_sample_ns


@pytest.mark.parametrize("failed", ["dds", "bridge"])
def test_source_and_bridge_fail_independently_then_recover(rig, failed):
    _, reader, exchange, fresh, _ = rig
    fresh()
    deadline = time.monotonic() + .65
    while time.monotonic() < deadline:
        exchange(dds=failed != "dds", bridge=failed != "bridge")
    for _ in range(3):
        assert exchange(dds=failed != "dds", bridge=failed != "bridge") is None
    assert "DDS" in reader.failure if failed == "dds" else "upstream" in reader.failure
    assert fresh() is not None


def test_tick_regression_stays_latched(rig):
    guard, reader, exchange, fresh, _ = rig
    fresh()
    guard.regress_tick()
    for _ in range(4):
        assert exchange() is None
    assert "tick regressed" in reader.failure


def test_guard_restart_requires_new_thor_reader(rig):
    guard, reader, exchange, fresh, _ = rig
    fresh()
    guard.close()
    replacement = LoopbackStateGuard(
        upstream_port=int(guard.profile["upstream_endpoint"].rsplit(":", 1)[1]),
        guard_port=guard.profile["guard_port"])
    try:
        # Feed the new attestor; exchange still publishes through the original bridge.
        for _ in range(5):
            replacement.observe(rig[4])
            assert exchange(dds=False) is None
        assert "restarted" in reader.failure
        new_reader = GuardedStateStream("127.0.0.1", guard.profile["guard_port"])
        try:
            replacement.observe(rig[4])
            assert new_reader.read(timeout_ms=200) is not None
        finally:
            new_reader.close()
    finally:
        replacement.close()


def test_operator_probe_wires_both_ends_and_keeps_default_optional():
    source = (ROOT / "tools/gb10/operator_probe.py").read_text()
    assert 'mock_options = ["--state-guard"] if args.state_guard else []' in source
    assert 'guard_options = ["--boundary-state-guard", "--boundary-state-port", "5558"] if args.state_guard else []' in source
    assert '"state_guard": args.state_guard' in source
    assert 'args.case == "guard-dds" and not args.state_guard' in source
    wrapper = (ROOT / "docker/venue_entry.sh").read_text()
    assert "--boundary-state-guard" not in wrapper
