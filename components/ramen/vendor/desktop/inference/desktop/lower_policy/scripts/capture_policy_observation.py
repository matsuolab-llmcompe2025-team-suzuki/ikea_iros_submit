"""Capture one read-only real observation for an isolated policy process.

The Unitree DDS runtime is Python 3.10 while the RAMEN-Ori model environment is
Python 3.12.  This tool keeps that ABI boundary explicit: it only subscribes to
joint, Dex1 and camera topics, assembles the 71D state, and writes an NPZ.  It
does not import or construct any actuator.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


ALLOWED_CAMERA_SHAPES = {
    "head_stereo": {(480, 1280, 3)},
    "head_left": {(480, 640, 3)},
    "head_right": {(480, 640, 3)},
    # D405 advertises 848x480 on the real G1.  Older recorded/adapted paths
    # may provide 640x480.  Preserve the native frame and let the policy's
    # single antialiased resize map it to 224x224.
    "wrist_left": {(480, 640, 3), (480, 848, 3)},
    "wrist_right": {(480, 640, 3), (480, 848, 3)},
}


def _split_head_bundle(bundle: dict[str, object]) -> dict[str, object]:
    """Turn one packed head subscription into two logical camera frames."""
    from inference.desktop.perception.frame_source import FrameData

    head = bundle["head_stereo"]
    if head.rgb.shape != (480, 1280, 3):
        raise ValueError(
            "head_stereo must be uint8 (480, 1280, 3), got "
            f"shape={head.rgb.shape} dtype={head.rgb.dtype}"
        )
    common = {
        "t": int(head.t),
        "received_monotonic_ns": _received_ns(head),
    }
    return {
        "head_left": FrameData(
            rgb=np.ascontiguousarray(head.rgb[:, :640]), **common
        ),
        "head_right": FrameData(
            rgb=np.ascontiguousarray(head.rgb[:, 640:]), **common
        ),
        "wrist_left": bundle["wrist_left"],
        "wrist_right": bundle["wrist_right"],
    }


def _complete_snapshot(sources: dict[str, object]):
    snapshots = {name: source.get() for name, source in sources.items()}
    if any(snapshot is None for snapshot in snapshots.values()):
        return None
    return snapshots


def _wait_for_camera_bundle(
    sources: dict[str, object],
    *,
    deadline: float,
    after_timestamps: dict[str, int] | None = None,
) -> dict[str, object]:
    while time.monotonic() < deadline:
        snapshots = _complete_snapshot(sources)
        if snapshots is not None:
            if after_timestamps is None or all(
                int(snapshots[name].t) != after_timestamps[name]
                for name in sources
            ):
                return snapshots
        time.sleep(0.005)
    missing = [name for name, source in sources.items() if source.get() is None]
    if missing:
        raise TimeoutError(f"camera frame timeout: missing={missing}")
    if after_timestamps is not None:
        latest = {name: int(source.get().t) for name, source in sources.items()}
        stalled = [
            name for name in sources if latest[name] == after_timestamps[name]
        ]
        raise TimeoutError(
            "camera streams did not all advance before timeout: "
            f"stalled={stalled} previous={after_timestamps} latest={latest}"
        )
    raise TimeoutError("camera streams did not all advance before timeout")


def _validate_frames(bundle: dict[str, object]) -> None:
    for role, snapshot in bundle.items():
        frame = snapshot.rgb
        allowed = ALLOWED_CAMERA_SHAPES[role]
        if frame.shape not in allowed or frame.dtype != np.uint8:
            raise ValueError(
                f"{role} must be uint8 with shape in {sorted(allowed)}, got "
                f"shape={frame.shape} dtype={frame.dtype}"
            )
        if not np.any(frame):
            raise ValueError(f"{role} is an all-zero image")


def _received_ns(snapshot: object) -> int:
    value = getattr(snapshot, "received_monotonic_ns", None)
    if value is None or int(value) <= 0:
        raise ValueError("camera frame has no valid host receive timestamp")
    return int(value)


def _measure_camera_rates(
    sources: dict[str, object],
    *,
    duration_s: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, int]]]:
    """Measure decoded latest-frame transition rate on the Desktop host."""
    diagnostics_before = {
        name: source.diagnostics() for name, source in sources.items()
    }
    transitions: dict[str, list[tuple[int, int]]] = {
        name: [] for name in sources
    }
    last_headers: dict[str, int | None] = {name: None for name in sources}
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        for name, source in sources.items():
            snapshot = source.get()
            if snapshot is None:
                continue
            header_ns = int(snapshot.t)
            if header_ns == last_headers[name]:
                continue
            transitions[name].append((header_ns, _received_ns(snapshot)))
            last_headers[name] = header_ns
        time.sleep(0.002)

    rates: dict[str, float] = {}
    for name, samples in transitions.items():
        if len(samples) < 2:
            rates[name] = 0.0
            continue
        elapsed_s = (samples[-1][1] - samples[0][1]) / 1e9
        rates[name] = (len(samples) - 1) / elapsed_s if elapsed_s > 0 else 0.0
    diagnostics_after = {
        name: source.diagnostics() for name, source in sources.items()
    }
    received_rates = {
        name: (
            diagnostics_after[name]["received_count"]
            - diagnostics_before[name]["received_count"]
        )
        / duration_s
        for name in sources
    }
    return rates, received_rates, diagnostics_after


def _wait_for_robot_state(joint_source, dex1_source, deadline: float):
    while time.monotonic() < deadline:
        joint = joint_source.get()
        dex1 = dex1_source.get()
        if (
            joint is not None
            and joint.position is not None
            and len(joint.position) == 29
            and dex1 is not None
        ):
            return joint, dex1
        time.sleep(0.005)
    raise TimeoutError("joint/Dex1 state timeout")


def _build_ramen_state(previous, current) -> np.ndarray:
    from inference.desktop.lower_policy.policies.base import RawRobotState
    from inference.desktop.lower_policy.policies.ramen_ori import build_state_from_raw
    from inference.desktop.perception.g1_urdf_fk import DEFAULT_URDF_PATH, G1WristFK

    urdf_path = Path(DEFAULT_URDF_PATH)
    if not urdf_path.exists():
        urdf_path = (Path(__file__).parents[4] / DEFAULT_URDF_PATH).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"G1 URDF not found: {urdf_path}")
    fk = G1WristFK.from_urdf(str(urdf_path))

    prev_joint, prev_dex1 = previous
    joint, dex1 = current
    prev_q = np.asarray(prev_joint.position, dtype=np.float32)
    q = np.asarray(joint.position, dtype=np.float32)
    prev_hand = prev_dex1.position_rad.astype(np.float32, copy=True)
    hand = dex1.position_rad.astype(np.float32, copy=True)
    previous_hold_target = np.concatenate([prev_q[12:29], prev_hand]).astype(
        np.float32, copy=False
    )
    raw = RawRobotState(
        joint_positions=q,
        hand_state=hand,
        ee_state=fk.compute_ee_state(q),
        last_action_19d=previous_hold_target,
        joint_positions_prev=prev_q,
        hand_state_prev=prev_hand,
    )
    state = build_state_from_raw(raw)
    if state.shape != (71,) or not np.all(np.isfinite(state)):
        raise ValueError("assembled RAMEN-Ori state must be finite 71D")
    return state


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture 4 real cameras + 71D state without robot commands."
    )
    parser.add_argument("--interface", required=True)
    parser.add_argument(
        "--head-topic",
        default="/head/camera/color/image_raw/compressed",
    )
    parser.add_argument(
        "--wrist-left-topic",
        default="/wrist_left/camera/color/image_raw/compressed",
    )
    parser.add_argument(
        "--wrist-right-topic",
        default="/wrist_right/camera/color/image_raw/compressed",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--rate-window-seconds", type=float, default=2.0)
    parser.add_argument("--min-camera-hz", type=float, default=25.0)
    parser.add_argument("--max-camera-skew-ms", type=float, default=50.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/smoke_observation_ramen.npz"),
    )
    args = parser.parse_args()
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    for name in (
        "rate_window_seconds",
        "min_camera_hz",
        "max_camera_skew_ms",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    from inference.desktop.perception.dex1_state_source import Dex1StateSource
    from inference.desktop.perception.frame_source import Ros2FrameSource
    from inference.desktop.perception.joint_state_source import JointStateSource

    ChannelFactoryInitialize(0, args.interface)
    # Subscribe to packed head exactly once.  Two subscriptions duplicate the
    # same 1280x480 JPEG over DDS and decode it twice, reducing all camera rates.
    physical_cameras = {
        "head_stereo": Ros2FrameSource(args.head_topic, stereo_view="packed"),
        "wrist_left": Ros2FrameSource(args.wrist_left_topic),
        "wrist_right": Ros2FrameSource(args.wrist_right_topic),
    }
    joint_source = JointStateSource()
    dex1_source = Dex1StateSource()
    try:
        deadline = time.monotonic() + args.timeout
        previous_physical = _wait_for_camera_bundle(
            physical_cameras, deadline=deadline
        )
        _validate_frames(previous_physical)
        previous_frames = _split_head_bundle(previous_physical)
        _validate_frames(previous_frames)
        previous_state = _wait_for_robot_state(joint_source, dex1_source, deadline)

        (
            camera_rates_hz,
            camera_receive_rates_hz,
            camera_diagnostics,
        ) = _measure_camera_rates(
            physical_cameras,
            duration_s=args.rate_window_seconds,
        )
        slow_roles = [
            role
            for role, rate in camera_rates_hz.items()
            if rate < args.min_camera_hz
        ]
        if slow_roles:
            raise RuntimeError(
                "camera decoded transition rate below threshold: "
                f"minimum={args.min_camera_hz:.2f}Hz rates={camera_rates_hz} "
                f"receive_rates={camera_receive_rates_hz} "
                f"diagnostics={camera_diagnostics} "
                f"slow={slow_roles}"
            )

        previous_timestamps = {
            name: int(snapshot.t)
            for name, snapshot in previous_physical.items()
        }
        # The rate window may intentionally be longer than --timeout.  Use a
        # fresh deadline for the post-measurement sample instead of reusing the
        # startup deadline, which has already expired by this point.
        deadline = time.monotonic() + args.timeout
        current_physical = _wait_for_camera_bundle(
            physical_cameras,
            deadline=deadline,
            after_timestamps=previous_timestamps,
        )
        _validate_frames(current_physical)
        current_frames = _split_head_bundle(current_physical)
        _validate_frames(current_frames)
        current_state = _wait_for_robot_state(joint_source, dex1_source, deadline)
        state = _build_ramen_state(previous_state, current_state)

        camera_header_timestamps = {
            name: int(snapshot.t)
            for name, snapshot in current_physical.items()
        }
        camera_receive_timestamps = {
            name: _received_ns(snapshot)
            for name, snapshot in current_physical.items()
        }
        camera_skew_ms = (
            max(camera_receive_timestamps.values())
            - min(camera_receive_timestamps.values())
        ) / 1e6
        if camera_skew_ms > args.max_camera_skew_ms:
            raise RuntimeError(
                "camera host-receive skew above threshold: "
                f"actual={camera_skew_ms:.2f}ms "
                f"maximum={args.max_camera_skew_ms:.2f}ms "
                f"receive_ns={camera_receive_timestamps}"
            )
        metadata = {
            "schema": "ramen_policy_observation_v2",
            "captured_monotonic_ns": time.monotonic_ns(),
            "interface": args.interface,
            "topics": {
                "head": args.head_topic,
                "wrist_left": args.wrist_left_topic,
                "wrist_right": args.wrist_right_topic,
            },
            "camera_header_timestamps_ns": camera_header_timestamps,
            "camera_receive_monotonic_ns": camera_receive_timestamps,
            "camera_timestamp_source": "desktop_dds_callback_monotonic",
            "camera_rates_hz": camera_rates_hz,
            "camera_receive_rates_hz": camera_receive_rates_hz,
            "camera_diagnostics": camera_diagnostics,
            "camera_shapes": {
                name: list(snapshot.rgb.shape)
                for name, snapshot in current_frames.items()
            },
            "camera_skew_ms": camera_skew_ms,
            "joint_timestamp_ns": int(current_state[0].t),
            "dex1_timestamp_ns": int(current_state[1].t),
            "actuator_constructed": False,
        }
        arrays = {"state": state}
        for role, snapshot in current_frames.items():
            arrays[role] = snapshot.rgb
        for role, snapshot in previous_frames.items():
            arrays[f"{role}_prev"] = snapshot.rgb
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, ensure_ascii=False)
        )
        _write_npz_atomic(args.output, arrays)
    finally:
        for source in physical_cameras.values():
            source.close()
        joint_source.close()
        dex1_source.close()

    print(
        f"observation-capture-ok output={args.output} "
        f"camera_skew_ms={camera_skew_ms:.2f} "
        f"camera_rates_hz={camera_rates_hz} "
        f"camera_receive_rates_hz={camera_receive_rates_hz} "
        f"state_dim={state.size} "
        "actuator=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
