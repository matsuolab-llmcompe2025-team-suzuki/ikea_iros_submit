"""実機 smoke: G1HandActuator (Phase B-3) real DDS dispatch (Issue #125 Phase 1.6)。

## 目的

`G1HandActuator` (unitree_sdk2py bundled Dex1 support、MotorCmds_ on
rt/dex1/{left,right}/cmd) を通じて Dex1-1 gripper が open/close 追従することを
目視 verify。

- Model は一切走らせない (Phase 1.6 = pipeline 単体 verify)
- 左右独立に指令 (`left:0.0,right:1.0` で片手だけ close の切り分け可能)
- 公式DDSとdatasetに一致する物理motor-output rad (0=閉側、約5.4=開側)
- `--execute`と確認語が無い限りpublisherを作らない

## 前提

- runtime env (`pixi run -e runtime`)
- Orin 側 `dex1_1_gripper_server.service` が起動 (Phase 1.1 で verify 済)
- Dex1-1 が USB で Orin に接続、hand の周辺に指 / 物を挟まない
- ChannelFactory init は自動 (arm_actuator 経由でなく本 script 単独で init)

## Sequence 文法

`--sequence` に "op:value,op:value,..." を渡す:

| op | value | 効果 |
|---|---|---|
| left | [0, 5.4] float | left gripper q [rad] (右手hold継続) |
| right | [0, 5.4] float | right gripper q [rad] (左手hold継続) |
| both | [0, 5.4] float | 左右同時に指定 q [rad] |
| pause | seconds (float) | 現 target で hold |

終了時は起動時に実測した左右位置へ戻す。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Sequence


# Unitree's bundled Dex1-1 test accepts |q-target| < 0.05 rad.  The Desktop
# subscriber and the 2 kHz serial bridge are sampled asynchronously, so allow
# 0.01 rad of observation margin instead of declaring a healthy gripper failed
# at the boundary.  At the documented 0.6 rad/cm conversion this is 1 mm.
RETURN_TOLERANCE_RAD = 0.06


@dataclass
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
        op, val = item.split(":", 1)
        op = op.strip().lower()
        if op not in {"left", "right", "both", "pause"}:
            raise ValueError(f"unknown op {op!r} in sequence item {item!r}")
        steps.append(_Step(op=op, value=float(val)))
    return steps


def _run(interface: str, steps: Sequence[_Step], *, execute: bool) -> int:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    from inference.desktop.lower_policy.actuators.hand import (
        G1_DEX1_LEFT_CMD_TOPIC,
        G1_DEX1_RIGHT_CMD_TOPIC,
        G1HandActuator,
        HAND_GRIP_MAX,
        HAND_GRIP_MIN,
    )
    from inference.desktop.perception.dex1_state_source import Dex1StateSource

    ChannelFactoryInitialize(0, interface)

    for step in steps:
        if step.op == "pause":
            if step.value < 0.0:
                raise ValueError("pause must be non-negative")
        elif not HAND_GRIP_MIN <= step.value <= HAND_GRIP_MAX:
            raise ValueError(
                f"{step.op} target must be within [{HAND_GRIP_MIN}, {HAND_GRIP_MAX}] rad"
            )

    state_source = Dex1StateSource()
    deadline = time.monotonic() + 5.0
    snapshot = None
    while snapshot is None and time.monotonic() < deadline:
        snapshot = state_source.get()
        time.sleep(0.01)
    if snapshot is None:
        state_source.close()
        raise RuntimeError("Dex1 bilateral state unavailable; no command was sent")
    initial = snapshot.position_rad.astype(float).tolist()
    print(
        f"[read-only] current Dex1 rad: left={initial[0]:.4f} right={initial[1]:.4f}",
        file=sys.stderr,
    )
    if not execute:
        print("[read-only] --execute not supplied; no publisher or command created")
        state_source.close()
        return 0

    confirmation = input(
        "Clear both grippers of fingers/objects. Type MOVE DEX1 to execute: "
    )
    if confirmation != "MOVE DEX1":
        print("[cancelled] confirmation did not match; no publisher or command created")
        state_source.close()
        return 1

    hand = G1HandActuator()
    print(
        "[smoke] G1HandActuator init: publishing to "
        f"{G1_DEX1_LEFT_CMD_TOPIC} + {G1_DEX1_RIGHT_CMD_TOPIC} @ 200 Hz",
        file=sys.stderr,
    )
    hand.start()
    print("[smoke] hand actuator started", file=sys.stderr)

    # The first command holds the measured pose; never jump to an assumed zero.
    cur = initial.copy()
    hand.send_action(cur)
    print(f"[smoke] initial hand q: left={cur[0]:.2f} right={cur[1]:.2f}", file=sys.stderr)

    return_ok = True
    try:
        for step in steps:
            if step.op == "pause":
                time.sleep(step.value)
                print(
                    f"[pause {step.value:g}s] current target: "
                    f"left={cur[0]:.2f} right={cur[1]:.2f}"
                )
                continue

            if step.op == "left":
                cur[0] = step.value
            elif step.op == "right":
                cur[1] = step.value
            elif step.op == "both":
                cur = [step.value, step.value]

            hand.send_action(cur)
            print(
                f"[send {step.op}:{step.value:g}] target: "
                f"left={cur[0]:.2f} right={cur[1]:.2f}"
            )
    except KeyboardInterrupt:
        print("[smoke] KeyboardInterrupt、returning to measured start pose", file=sys.stderr)
    finally:
        try:
            hand.send_action(initial)
            return_deadline = time.monotonic() + 3.0
            return_error = float("inf")
            returned = None
            while time.monotonic() < return_deadline:
                returned = state_source.get()
                if returned is not None:
                    actual = returned.position_rad.astype(float)
                    return_error = max(
                        abs(actual[0] - initial[0]),
                        abs(actual[1] - initial[1]),
                    )
                    if return_error <= RETURN_TOLERANCE_RAD:
                        break
                time.sleep(0.01)
            if returned is None:
                return_ok = False
                print(
                    "[smoke] WARNING: no Dex1 feedback while returning to start pose",
                    file=sys.stderr,
                )
            else:
                actual = returned.position_rad.astype(float)
                status = (
                    "reached"
                    if return_error <= RETURN_TOLERANCE_RAD
                    else "TIMEOUT"
                )
                return_ok = return_error <= RETURN_TOLERANCE_RAD
                print(
                    f"[smoke] return {status}: left={actual[0]:.4f} "
                    f"right={actual[1]:.4f} max_error={return_error:.4f}rad",
                    file=sys.stderr,
                )
        except Exception as e:
            return_ok = False
            print(f"[smoke] measured-start return failed: {e}", file=sys.stderr)
        hand.stop()
        state_source.close()
        print("[smoke] hand actuator stopped", file=sys.stderr)
    return 0 if return_ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="G1HandActuator real DDS dispatch smoke (Issue #125 Phase 1.6)."
    )
    parser.add_argument("--interface", required=True, help="CycloneDDS NIC name")
    parser.add_argument(
        "--sequence",
        required=True,
        help=(
            "op:value,op:value,... sequence。ops: left / right / both / pause。"
            " 値は物理rad。最初はread-onlyの pause:2 で現在値を確認すること。"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="実機へ送信する。未指定時は現在値とsequence検証のみ。",
    )
    args = parser.parse_args()
    steps = _parse_sequence(args.sequence)
    return _run(args.interface, steps, execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
