"""Small, reversible waist smoke test for a physical G1 in Regular Mode.

No walking API or model is used. Targets are small offsets from the measured
waist pose, the 14 arm joints are held at their measured pose, and ownership is
returned with the official arm_sdk weight ramp before strict Regular is checked.
Without ``--execute`` this command is read-only.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np


MAX_OFFSET_RAD = 0.10
POSE_TOLERANCE_RAD = 0.04


@dataclass(frozen=True)
class _Step:
    op: str
    value: float


def _parse_sequence(raw: str) -> list[_Step]:
    steps: list[_Step] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"sequence item {item!r} missing ':'")
        op, value_raw = item.split(":", 1)
        op = op.strip().lower()
        if op not in {"yaw", "roll", "pitch", "zero", "pause"}:
            raise ValueError(f"unknown op {op!r} in sequence item {item!r}")
        value = float(value_raw)
        if not np.isfinite(value):
            raise ValueError(f"non-finite value in sequence item {item!r}")
        if op == "pause" and value <= 0.0:
            raise ValueError("pause duration must be positive")
        if op in {"yaw", "roll", "pitch"} and abs(value) > MAX_OFFSET_RAD:
            raise ValueError(
                f"{op} offset exceeds diagnostic limit {MAX_OFFSET_RAD:.2f} rad"
            )
        if op == "zero" and value != 0.0:
            raise ValueError("zero step requires value 0")
        steps.append(_Step(op=op, value=value))
    if not steps:
        raise ValueError("sequence must contain at least one step")
    return steps


def _target_for_step(initial: np.ndarray, step: _Step) -> np.ndarray:
    """Return a one-axis offset target; ``zero`` means measured initial pose."""
    target = np.asarray(initial, dtype=np.float64).copy()
    if target.shape != (3,):
        raise ValueError(f"initial waist pose must have shape (3,), got {target.shape}")
    if step.op == "pause":
        raise ValueError("pause has no target")
    if step.op == "zero":
        return target
    index = {"yaw": 0, "roll": 1, "pitch": 2}[step.op]
    target[index] += step.value
    return target


def _ramp_waist(
    waist: object,
    start: np.ndarray,
    target: np.ndarray,
    *,
    duration_s: float,
    update_hz: float = 50.0,
) -> None:
    steps = max(1, int(round(duration_s * update_hz)))
    sleep_s = duration_s / steps
    for index in range(1, steps + 1):
        alpha = index / steps
        waist.send_action(start + alpha * (target - start))  # type: ignore[attr-defined]
        time.sleep(sleep_s)


def _wait_for_waist(
    arm: object,
    target: np.ndarray,
    *,
    timeout_s: float,
) -> tuple[bool, np.ndarray, float]:
    deadline = time.monotonic() + timeout_s
    actual = np.asarray(arm.read_waist_positions(), dtype=np.float64)  # type: ignore[attr-defined]
    error = float(np.max(np.abs(actual - target)))
    while time.monotonic() < deadline:
        actual = np.asarray(arm.read_waist_positions(), dtype=np.float64)  # type: ignore[attr-defined]
        error = float(np.max(np.abs(actual - target)))
        if error <= POSE_TOLERANCE_RAD:
            return True, actual, error
        time.sleep(0.01)
    return False, actual, error


def _run(
    interface: str,
    steps: Sequence[_Step],
    *,
    execute: bool,
    ramp_seconds: float,
) -> int:
    from inference.desktop.lower_policy.actuators.g1_arm_sdk import G1ArmActuator
    from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator
    from inference.desktop.lower_policy.actuators.waist import G1WaistActuator

    # Read-only high-level check. No locomotion method is called.
    loco = G1SDKWalkActuator(interface=interface)
    status = loco.get_loco_status()
    if (status.fsm_id, status.fsm_mode) != (501, 0):
        raise RuntimeError(
            "G1 must be in strict Regular before waist smoke: "
            f"actual=({status.fsm_id},{status.fsm_mode}), expected=(501,0)"
        )

    arm = G1ArmActuator()
    initial_waist = np.asarray(arm.read_waist_positions(), dtype=np.float64)
    hold_arm = np.asarray(arm.read_arm_positions(), dtype=np.float64)
    if initial_waist.shape != (3,) or not np.all(np.isfinite(initial_waist)):
        raise RuntimeError(f"invalid measured waist pose: {initial_waist!r}")
    if hold_arm.shape != (14,) or not np.all(np.isfinite(hold_arm)):
        raise RuntimeError(f"invalid measured arm pose: shape={hold_arm.shape}")

    print(
        "[read-only] strict Regular=(501,0); measured waist "
        f"yaw={initial_waist[0]:+.3f} roll={initial_waist[1]:+.3f} "
        f"pitch={initial_waist[2]:+.3f}rad"
    )
    if not execute:
        print("[read-only] no --execute; no robot command was sent")
        return 0

    confirmation = input(
        "Harness / E-stop confirmed. Clear the robot and type MOVE WAIST to "
        "run small relative waist offsets and return: "
    )
    if confirmation.strip() != "MOVE WAIST":
        raise RuntimeError("confirmation mismatch; no robot command was sent")

    # Re-check immediately before the first command.
    status = loco.get_loco_status()
    if (status.fsm_id, status.fsm_mode) != (501, 0):
        raise RuntimeError(
            "Regular changed before actuation; no robot command was sent: "
            f"actual=({status.fsm_id},{status.fsm_mode})"
        )

    waist = G1WaistActuator(arm)
    arm.send_action(hold_arm)
    arm.send_waist_action(initial_waist)
    arm.start()
    print("[run] arm_sdk started; arms hold measured pose; no walking command")
    target_ok = True
    return_ok = False
    current_target = initial_waist.copy()
    try:
        time.sleep(0.05)
        for step in steps:
            if step.op == "pause":
                time.sleep(step.value)
                actual = np.asarray(arm.read_waist_positions(), dtype=np.float64)
                error = float(np.max(np.abs(actual - current_target)))
                print(
                    f"[pause {step.value:g}s] target={current_target.tolist()} "
                    f"actual={actual.tolist()} max_error={error:.4f}rad"
                )
                continue

            target = _target_for_step(initial_waist, step)
            _ramp_waist(
                waist,
                current_target,
                target,
                duration_s=ramp_seconds,
            )
            reached, actual, error = _wait_for_waist(
                arm, target, timeout_s=3.0
            )
            print(
                f"[step {step.op}:{step.value:+.3f}] "
                f"{'reached' if reached else 'TIMEOUT'} actual={actual.tolist()} "
                f"max_error={error:.4f}rad"
            )
            current_target = target
            if not reached:
                target_ok = False
                break
    except KeyboardInterrupt:
        target_ok = False
        print("[stop] interrupted; returning to measured initial waist pose")
    finally:
        actual = np.asarray(arm.read_waist_positions(), dtype=np.float64)
        _ramp_waist(
            waist,
            actual,
            initial_waist,
            duration_s=ramp_seconds,
        )
        returned, _, error = _wait_for_waist(
            arm, initial_waist, timeout_s=5.0
        )
        return_ok = returned
        print(
            f"[stop] waist return {'reached' if returned else 'TIMEOUT'} "
            f"max_error={error:.4f}rad"
        )
        try:
            print("[stop] arm_sdk controlled release started (weight 1.00 -> 0.00)")
            arm.controlled_release(duration_s=2.0)
            print("[stop] arm_sdk controlled release complete (weight=0.00)")
        finally:
            arm.stop()

    time.sleep(0.5)
    final_status = loco.get_loco_status()
    handoff_ok = (final_status.fsm_id, final_status.fsm_mode) == (501, 0)
    print(
        "[stop] strict Regular handoff "
        f"{'verified' if handoff_ok else 'FAILED'} "
        f"({final_status.fsm_id},{final_status.fsm_mode})"
    )
    return 0 if target_ok and return_ok and handoff_ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True, help="CycloneDDS NIC name")
    parser.add_argument(
        "--sequence",
        required=True,
        help=(
            "relative op:value sequence; yaw/roll/pitch max +/-0.10 rad, "
            "zero:0 returns to measured initial pose"
        ),
    )
    parser.add_argument(
        "--ramp-seconds",
        type=float,
        default=1.0,
        help="interpolation duration per target [s] (default: 1.0)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="enable physical motion after the exact interactive confirmation",
    )
    args = parser.parse_args()
    if not np.isfinite(args.ramp_seconds) or args.ramp_seconds <= 0.0:
        parser.error("--ramp-seconds must be finite and positive")
    return _run(
        args.interface,
        _parse_sequence(args.sequence),
        execute=args.execute,
        ramp_seconds=args.ramp_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
