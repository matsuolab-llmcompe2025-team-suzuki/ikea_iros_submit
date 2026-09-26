#!/usr/bin/env python3
"""Run real checkpoint inference without creating an action publisher.

Run inside the pinned image, with cwd=/app/ramen and its runtime Python.
Synthetic RGB tests execution and tensor contracts, not task performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml


def validate_action(action, dimension: int) -> np.ndarray:
    chunk = np.asarray(action.action_chunk)
    if chunk.ndim != 2 or not len(chunk) or chunk.shape[1] != dimension:
        raise ValueError(f"Unexpected action shape {chunk.shape}, expected (T,{dimension})")
    if not np.isfinite(chunk).all():
        raise ValueError("Non-finite action")
    if not np.isfinite(action.latency_ms) or action.latency_ms < 0:
        raise ValueError("Invalid inference latency")
    return chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    sys.path.insert(0, str(Path.cwd()))
    from inference.desktop import assembly
    from inference.desktop.lower_policy.initial_pose import apply_policy_variant_profile, initial_pose_from_config
    from inference.desktop.lower_policy.policies.base import CameraKey, Observation, RawRobotState
    from inference.desktop.lower_policy.policies.config_loader import load_policy_variant

    config_path = Path("inference/desktop/lower_policy/configs/policy_config.yaml")
    skill_config = yaml.safe_load(Path("inference/desktop/lower_policy/configs/skill_config.yaml").read_text())
    entry = load_policy_variant(config_path, args.variant)
    skill_config = apply_policy_variant_profile(skill_config, args.skill, args.variant)
    pose = initial_pose_from_config(skill_config, args.skill)
    fk = assembly.FkFactory().for_skill(skill_config, args.skill)
    if fk is None:
        raise RuntimeError("Missing FK; do not silently replace EEF with zeros")
    q = np.zeros(29, dtype=np.float32)
    q[15:29] = pose.arm_position_rad
    raw = RawRobotState(
        joint_positions=q,
        hand_state=np.asarray(pose.dex1_target_rad, dtype=np.float32),
        ee_state=fk.compute_ee_state(q),
    )
    # Workers decode their native action representation to the common upper-
    # body contract: waist3 + arms14 + Dex1 motor angles2.
    expected_dim = 19
    started = time.monotonic()
    policy = assembly.load_policy(entry)
    load_sec = time.monotonic() - started
    perception = None
    records, targets = [], []
    try:
        if entry.policy_config.mode != "none":
            perception = assembly.build_yolo_perception(config_path)
        previous = None
        for tick in range(args.steps):
            tick_start = time.monotonic()
            reset = tick == args.steps // 2 and callable(getattr(policy, "reset", None))
            if reset:
                policy.reset()
                previous = None
            frames = {}
            for i, camera in enumerate(CameraKey):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :, 0] = np.arange(640, dtype=np.uint16)[None, :] % 256
                frame[:, :, 1] = np.arange(480, dtype=np.uint16)[:, None] % 256
                frame[:, :, 2] = (tick * 3 + i * 47) % 256
                frames[camera] = frame
            detections = None
            if perception is not None:
                detections = {cam: perception.predict(frames[cam])
                              for cam in (CameraKey.HEAD_LEFT, CameraKey.HEAD_RIGHT)}
            state = policy.build_state_from_raw(raw)
            if not np.isfinite(state).all():
                raise ValueError("Non-finite assembled state")
            observation = Observation(
                frames_bgr=frames, frames_bgr_prev=previous, state=state,
                skill_id=entry.policy_config.skill_id,
                language=entry.policy_config.language_prompt or args.skill.replace("_", " "),
                obb_detections=detections, timestamp_ns=time.monotonic_ns(),
            )
            action = policy.predict(observation)
            chunk = validate_action(action, expected_dim)
            targets.append(chunk[0].copy())
            record = {"tick": tick, "reset": reset, "state_shape": list(state.shape),
                      "action_shape": list(chunk.shape), "latency_ms": float(action.latency_ms),
                      "wall_ms": (time.monotonic() - tick_start) * 1000,
                      "action_min": float(chunk.min()), "action_max": float(chunk.max())}
            records.append(record)
            print(json.dumps(record), flush=True)
            previous = frames
            time.sleep(max(0, 1 / 30 - (time.monotonic() - tick_start)))
    finally:
        policy.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), targets=np.asarray(targets))
    result = {"variant": args.variant, "skill": args.skill, "input": "synthetic RGB / fixed FK pose",
              "physical_commands_sent": False, "task_success_evaluated": False,
              "load_sec": load_sec, "passed": True, "records": records}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
