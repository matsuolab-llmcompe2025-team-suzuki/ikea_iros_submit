"""実機 smoke: state assembly + URDF FK verify (Issue #125 Phase 1.7 + 1.8)。

## 目的

Phase 1.7 (FK verify) と Phase 1.8 (state slice range) を model 抜きで走らせる。
- fk_only mode: joint_state 受信 → G1WristFK → ee_state を dump (Phase 1.7)
- ramen_ori_71d mode: 30 tick loop で build_state_from_raw (RAMEN-Ori 71D) を
  回し、全 slice の統計 (min/max/mean/finite ratio) を dump (Phase 1.8)
- groot_49d mode: 30 tick loop で GR00T 49D 版

## 前提

- runtime env (`pixi run -e runtime`)
- Orin real_hw_bridge_node が `/joint_states` を publish (Phase 1.2 で verify 済)
- ChannelFactory init は自動で行う (`ChannelFactoryInitialize(0, interface)`)。

## 使い方

```bash
pixi run -e runtime python -m inference.desktop.lower_policy.scripts.smoke_state_assembly \\
    --interface eth0 --mode fk_only --dump-path /tmp/smoke_fk.json

pixi run -e runtime python -m inference.desktop.lower_policy.scripts.smoke_state_assembly \\
    --interface eth0 --mode ramen_ori_71d --ticks 30 --dump-path /tmp/smoke_state_ramen.json

pixi run -e runtime python -m inference.desktop.lower_policy.scripts.smoke_state_assembly \\
    --interface eth0 --mode groot_49d --ticks 30 --dump-path /tmp/smoke_state_groot.json
```

## Model 不要

policy は一切 load しない。build_state_from_raw の numpy 実装のみ検証。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _summarize_slice(name: str, arr: np.ndarray, expected_dim: int) -> dict:
    """slice の統計 dict を返す (min/max/mean/finite ratio + shape verify)。"""
    if arr.shape != (expected_dim,):
        return {"name": name, "error": f"shape mismatch {arr.shape} != ({expected_dim},)"}
    finite = np.isfinite(arr)
    return {
        "name": name,
        "shape": list(arr.shape),
        "min": float(arr.min()) if finite.all() else None,
        "max": float(arr.max()) if finite.all() else None,
        "mean": float(arr.mean()) if finite.all() else None,
        "finite_ratio": float(finite.mean()),
    }


def _run_fk_only(interface: str, tick_seconds: float = 3.0) -> dict:
    """Phase 1.7: 静止した joint_state → G1WristFK → ee_state dump。"""
    # lazy import: SDK 依存を top-level で持たない (default env でも collect 通す)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    from inference.desktop.perception.g1_urdf_fk import (
        DEFAULT_URDF_PATH,
        G1WristFK,
    )
    from inference.desktop.perception.joint_state_source import JointStateSource

    ChannelFactoryInitialize(0, interface)
    # URDF path: repo root 相対 (entrypoint.py と同じ resolve pattern)
    urdf_path = Path(DEFAULT_URDF_PATH)
    if not urdf_path.exists():
        urdf_path = (Path(__file__).parents[4] / DEFAULT_URDF_PATH).resolve()
    if not urdf_path.exists():
        print(f"[smoke] URDF not found at {DEFAULT_URDF_PATH}", file=sys.stderr)
        return {"error": "urdf_missing"}
    fk = G1WristFK.from_urdf(str(urdf_path))
    print(f"[smoke] URDF FK loaded from {urdf_path}", file=sys.stderr)

    src = JointStateSource(topic="/joint_states")
    print(f"[smoke] JointStateSource init、waiting {tick_seconds}s for lowstate sync...", file=sys.stderr)
    time.sleep(tick_seconds)

    snap = src.get()
    src.close()
    if snap is None:
        return {"error": "no_joint_state_received"}
    if snap.position is None or len(snap.position) != 29:
        return {
            "error": "joint_state_wrong_dim",
            "received_dim": len(snap.position) if snap.position is not None else None,
        }

    jp = np.asarray(snap.position, dtype=np.float32)
    ee_state = fk.compute_ee_state(jp)
    result = {
        "mode": "fk_only",
        "joint_state_ts_ns": int(snap.t),
        "joint_positions": jp.tolist(),
        "joint_summary": _summarize_slice("joint_positions", jp, 29),
        "ee_state": ee_state.tolist(),
        "ee_left_xyz": ee_state[0:3].tolist(),
        "ee_left_euler": ee_state[3:6].tolist(),
        "ee_right_xyz": ee_state[6:9].tolist(),
        "ee_right_euler": ee_state[9:12].tolist(),
        "ee_summary": _summarize_slice("ee_state", ee_state, 12),
    }
    return result


def _run_state_loop(
    interface: str, mode: str, ticks: int, tick_hz: float
) -> dict:
    """Phase 1.8: 30 tick 相当の loop で build_state_from_raw を回す + 統計 dump。

    Mode:
        - ramen_ori_71d: RAMEN-Ori の build_state_from_raw
        - groot_49d: GR00T の build_state_from_raw
    """
    # lazy import
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    from inference.desktop.lower_policy.policies.base import RawRobotState
    from inference.desktop.perception.g1_urdf_fk import (
        DEFAULT_URDF_PATH,
        G1WristFK,
    )
    from inference.desktop.perception.dex1_state_source import Dex1StateSource
    from inference.desktop.perception.joint_state_source import JointStateSource

    if mode == "ramen_ori_71d":
        from inference.desktop.lower_policy.policies.ramen_ori import (
            build_state_from_raw,
        )
        expected_dim = 71
        slice_defs = [
            ("joint", slice(0, 19), 19),
            ("tracking_err", slice(19, 38), 19),
            ("velocity", slice(38, 57), 19),
            ("hand_state", slice(57, 59), 2),
            ("ee_pose", slice(59, 71), 12),
        ]
    elif mode == "groot_49d":
        from inference.desktop.lower_policy.policies.groot import (
            build_state_from_raw,
        )
        expected_dim = 49
        slice_defs = [
            ("left_wrist_eef_9d", slice(0, 9), 9),
            ("right_wrist_eef_9d", slice(9, 18), 9),
            ("left_hand", slice(18, 25), 7),
            ("right_hand", slice(25, 32), 7),
            ("left_arm", slice(32, 39), 7),
            ("right_arm", slice(39, 46), 7),
            ("waist", slice(46, 49), 3),
        ]
    else:
        raise ValueError(f"unknown mode: {mode}")

    ChannelFactoryInitialize(0, interface)
    urdf_path = Path(DEFAULT_URDF_PATH)
    if not urdf_path.exists():
        urdf_path = (Path(__file__).parents[4] / DEFAULT_URDF_PATH).resolve()
    fk = G1WristFK.from_urdf(str(urdf_path)) if urdf_path.exists() else None
    if fk is None:
        print(f"[smoke] WARNING: URDF not found, ee_state = zeros", file=sys.stderr)

    src = JointStateSource(topic="/joint_states")
    dex1_src = Dex1StateSource()
    print(
        "[smoke] JointStateSource + read-only Dex1StateSource init、warmup 1s...",
        file=sys.stderr,
    )
    time.sleep(1.0)

    # tick loop with buffer (VlaSkill と同 pattern)
    prev_joint_positions: np.ndarray | None = None
    prev_hand_state: np.ndarray | None = None
    prev_action_19d: np.ndarray | None = None  # ramen_ori のみ意味あるが GR00T でも None

    per_tick_states = []
    per_tick_finite = []
    interval = 1.0 / max(tick_hz, 1.0)

    for i in range(ticks):
        t_start = time.monotonic()
        snap = src.get()
        if snap is None or snap.position is None or len(snap.position) != 29:
            print(f"[tick {i}] joint_state not ready (retry)", file=sys.stderr)
            time.sleep(interval)
            continue
        jp = np.asarray(snap.position, dtype=np.float32)
        hand_snap = dex1_src.get()
        if hand_snap is None:
            print(f"[tick {i}] Dex1 state not ready (retry)", file=sys.stderr)
            time.sleep(interval)
            continue
        hand_position_rad = hand_snap.position_rad
        ee_state = (
            fk.compute_ee_state(jp) if fk is not None else np.zeros(12, dtype=np.float32)
        )
        raw = RawRobotState(
            joint_positions=jp,
            hand_state=hand_position_rad,
            ee_state=ee_state,
            last_action_19d=prev_action_19d,
            joint_positions_prev=prev_joint_positions,
            hand_state_prev=prev_hand_state,
        )
        state = build_state_from_raw(raw)
        assert state.shape == (expected_dim,), f"tick {i}: unexpected state shape {state.shape}"
        per_tick_states.append(state)
        per_tick_finite.append(bool(np.all(np.isfinite(state))))

        # buffer update
        prev_joint_positions = jp
        prev_hand_state = hand_position_rad.copy()
        # Model無しのread-only smokeでは「現在姿勢をholdするtarget」を前actionと
        # みなす。all-zero commandを捏造すると、開いているDex1だけで約5.4 radの
        # 偽tracking errorになり、状態契約の診断を誤らせる。
        prev_action_19d = np.concatenate(
            [jp[12:29], hand_position_rad]
        ).astype(np.float32, copy=False)

        elapsed = time.monotonic() - t_start
        if elapsed < interval:
            time.sleep(interval - elapsed)
    src.close()
    dex1_src.close()

    if not per_tick_states:
        return {"error": "no_valid_ticks"}

    all_states = np.stack(per_tick_states, axis=0)  # (T, D)
    all_finite_ratio = float(sum(per_tick_finite) / len(per_tick_finite))

    slice_stats = []
    for name, sl, dim in slice_defs:
        # 全 tick の slice を統合して統計
        merged = all_states[:, sl].reshape(-1)
        slice_stats.append(_summarize_slice(name, merged, len(merged)))
        slice_stats[-1]["per_tick_shape"] = dim

    result = {
        "mode": mode,
        "ticks_completed": len(per_tick_states),
        "state_dim": expected_dim,
        "hand_state_unit": "dex1_motor_output_rad",
        "all_finite_ratio": all_finite_ratio,
        "state_slices": slice_stats,
        "first_tick_state": all_states[0].tolist(),
        "last_tick_state": all_states[-1].tolist(),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real robot state assembly + FK smoke (Issue #125 Phase 1.7 / 1.8)."
    )
    parser.add_argument("--interface", required=True, help="CycloneDDS NIC name (e.g. eth0)")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["fk_only", "ramen_ori_71d", "groot_49d"],
        help="fk_only=Phase 1.7 単発 FK dump、ramen_ori_71d/groot_49d=Phase 1.8 30-tick loop",
    )
    parser.add_argument("--ticks", type=int, default=30, help="tick 数 (state loop mode)")
    parser.add_argument("--tick-hz", type=float, default=30.0, help="tick rate [Hz]")
    parser.add_argument(
        "--dump-path", type=Path, default=None,
        help="結果 JSON dump path (省略時は stdout のみ)",
    )
    args = parser.parse_args()

    if args.mode == "fk_only":
        result = _run_fk_only(args.interface)
    else:
        result = _run_state_loop(args.interface, args.mode, args.ticks, args.tick_hz)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.dump_path is not None:
        args.dump_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[smoke] dumped to {args.dump_path}", file=sys.stderr)
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
