"""実機 smoke: real ckpt load + dry-run forward (Issue #125 Phase 1.9 + 1.10)。

## 目的

- **load-only mode (Phase 1.9)**: policy variant を指定して from_ckpt 成功 + warmup
  1 tick 走破。VRAM 使用量 / warmup latency 記録。chunk 内容の quality は問わない。
- **dry-run mode (Phase 1.10)**: 実 JointStateSource + G1WristFK で state を組み、
  30 tick loop で `Policy.predict` を回して chunk shape/latency/finiteness を dump。
  Mock actuator (log 保持のみ) で受信、physical 変化なし。

## Actuator は全 Mock

本 script は Model chunk が pipeline を流れることのみ verify、physical dispatch は
しない。real actuator verify は smoke_waist_real / smoke_hand_real + Phase 1-E の
entrypoint 起動で別途行う。

## 前提

- **desktop-inference env** (`cd inference/desktop && pixi run ...`)、torch +
  lerobot + hydra が入っている想定 (前 commit `314f434` で env 整備済)。
- 実 ckpt が HF に push 済 or local に配置済 (policy_config.yaml の ckpt_ref)。
- Orin real_hw_bridge_node が /joint_states publish (Phase 1.2 で verify 済、
  dry-run mode で使う)。
- runtime env (SDK + cyclonedds) はここでは不要 (Mock actuator 経路)、
  ただし dry-run mode の JointStateSource は cyclonedds 必要 → desktop-inference
  env に cyclonedds を足すか、runtime env に torch/lerobot を足すか、環境
  compat 判断が要る。**現状は desktop-inference env + cyclonedds 追加 install
  想定**。互換が取れない場合は state を zeros で回す --no-real-state flag も用意。

## 使い方

```bash
# Phase 1.9: 単に load + warmup 1 tick
cd inference/desktop && pixi run python -m inference.desktop.lower_policy.scripts.smoke_policy_forward \\
    --variant ramen_ori_default --mode load-only

# Phase 1.10: 30 tick dry-run (実 joint_state 入力、Mock actuator)
pixi run python -m inference.desktop.lower_policy.scripts.smoke_policy_forward \\
    --interface eth0 --variant ramen_ori_default --mode dry-run \\
    --ticks 30 --dump-path /tmp/smoke_forward_ramen.json

# 実 joint_state 未接続の debug 経路 (state=zeros で forward だけ verify)
pixi run python -m inference.desktop.lower_policy.scripts.smoke_policy_forward \\
    --variant ramen_ori_default --mode dry-run --no-real-state --ticks 5
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _default_config_path() -> Path:
    """policy_config.yaml の repo-relative default。"""
    return Path("inference/desktop/lower_policy/configs/policy_config.yaml")


def _load_policy(variant: str, config_path: Path):
    """variant → (VariantEntry, loaded Policy instance)。"""
    from inference.desktop.lower_policy.policies.config_loader import (
        load_policy_variant,
        resolve_policy_class,
    )

    entry = load_policy_variant(config_path, variant)
    policy_cls = resolve_policy_class(entry.policy_type)
    print(
        f"[smoke] variant={entry.name} type={entry.policy_type} "
        f"mode={entry.policy_config.mode} ckpt={entry.policy_config.ckpt_ref}",
        file=sys.stderr,
    )
    t0 = time.monotonic()
    policy = policy_cls.from_ckpt(entry.policy_config)
    dt_load = time.monotonic() - t0
    print(f"[smoke] ckpt loaded in {dt_load:.2f}s", file=sys.stderr)
    return entry, policy, dt_load


def _vram_snapshot() -> dict:
    """CUDA VRAM 使用量 (available なら)。"""
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
    except ImportError:
        return {"available": False, "reason": "torch import failed"}


def _build_zeros_observation(policy, entry) -> Any:
    """Zero-filled Observation (state / frames / obb すべて 0)、warmup 用途。"""
    from inference.desktop.lower_policy.policies.base import Observation

    H, W = 480, 640
    cameras = policy.CAMERAS
    frames = {cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in cameras}
    state = np.zeros(policy.STATE_DIM, dtype=np.float32)
    # skill / language: variant によって使う方が異なる
    skill_id = 0  # 適当な有効 id (0..NUM_SKILLS-1)
    # groot_pick_legs は worker が task を manifest task と厳密一致検証するので、
    # policy が既定 prompt を持つ場合はそれを使う (それ以外は任意の smoke 文字列)。
    language = getattr(policy, "DEFAULT_LANGUAGE_PROMPT", "smoke test")
    return Observation(
        frames_bgr=frames,
        frames_bgr_prev=None,
        state=state,
        skill_id=skill_id,
        language=language,
        obb_detections=None,
        timestamp_ns=time.monotonic_ns(),
    )


def _run_load_only(variant: str, config_path: Path, warmup_iters: int) -> dict:
    entry, policy, dt_load = _load_policy(variant, config_path)
    vram_after_load = _vram_snapshot()

    dummy_obs = _build_zeros_observation(policy, entry)
    t0 = time.monotonic()
    policy.warmup(n_iter=warmup_iters)
    dt_warmup = time.monotonic() - t0
    vram_after_warmup = _vram_snapshot()

    # 1 tick predict で shape verify
    t1 = time.monotonic()
    action = policy.predict(dummy_obs)
    dt_predict = time.monotonic() - t1

    result = {
        "mode": "load-only",
        "variant": variant,
        "policy_type": entry.policy_type,
        "load_seconds": dt_load,
        "warmup_seconds": dt_warmup,
        "warmup_iters": warmup_iters,
        "single_predict_ms": dt_predict * 1000,
        "action_chunk_shape": list(action.action_chunk.shape),
        "action_chunk_dtype": str(action.action_chunk.dtype),
        "action_chunk_finite": bool(np.all(np.isfinite(action.action_chunk))),
        "action_chunk_min": float(action.action_chunk.min()),
        "action_chunk_max": float(action.action_chunk.max()),
        "action_latency_ms": action.latency_ms,
        "vram_after_load": vram_after_load,
        "vram_after_warmup": vram_after_warmup,
    }
    policy.close()
    return result


def _get_joint_state(interface: str, timeout_s: float = 5.0):
    """cyclonedds init + JointStateSource で 1 snapshot 取る (dry-run 実 state 用)。"""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    from inference.desktop.perception.dex1_state_source import Dex1StateSource
    from inference.desktop.perception.joint_state_source import JointStateSource

    ChannelFactoryInitialize(0, interface)
    src = JointStateSource(topic="/joint_states")
    return src, Dex1StateSource()


def _load_captured_state(path: Path, expected_dim: int) -> np.ndarray:
    """`smoke_state_assembly` JSONから実測stateを安全に復元する。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") not in {"ramen_ori_71d", "groot_49d"}:
        raise ValueError(
            f"unsupported state dump mode in {path}: {payload.get('mode')!r}"
        )
    raw = payload.get("last_tick_state")
    if not isinstance(raw, list):
        raise ValueError(f"state dump has no last_tick_state list: {path}")
    state = np.asarray(raw, dtype=np.float32)
    if state.shape != (expected_dim,):
        raise ValueError(
            f"captured state shape mismatch: {state.shape} != ({expected_dim},)"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"captured state contains NaN/Inf: {path}")
    return state


def _load_captured_observation(path: Path, policy):
    """Read the command-free NPZ produced by capture_policy_observation."""
    from inference.desktop.lower_policy.policies.base import CameraKey
    from inference.desktop.perception.frame_source import (
        adapt_wrist_to_training_shape,
    )

    with np.load(path, allow_pickle=False) as payload:
        state = np.asarray(payload["state"], dtype=np.float32)
        if state.shape != (policy.STATE_DIM,) or not np.all(np.isfinite(state)):
            raise ValueError(
                f"observation state must be finite ({policy.STATE_DIM},), got {state.shape}"
            )
        frames = {}
        frames_prev = {}
        for camera in policy.CAMERAS:
            if not isinstance(camera, CameraKey):
                raise TypeError(f"unexpected camera key: {camera!r}")
            key = camera.value
            previous_key = f"{key}_prev"
            if key not in payload or previous_key not in payload:
                raise ValueError(f"observation dump is missing {key}/{previous_key}")
            current = np.asarray(payload[key])
            previous = np.asarray(payload[previous_key])
            allowed_shapes = (
                {(480, 640, 3), (480, 848, 3)}
                if camera.is_wrist
                else {(480, 640, 3)}
            )
            for label, frame in ((key, current), (previous_key, previous)):
                if (
                    frame.shape not in allowed_shapes
                    or frame.dtype != np.uint8
                    or not np.any(frame)
                ):
                    raise ValueError(
                        f"{label} must be nonzero uint8 with shape in "
                        f"{sorted(allowed_shapes)}, "
                        f"got shape={frame.shape} dtype={frame.dtype}"
                    )
            if camera.is_wrist:
                current = adapt_wrist_to_training_shape(current)
                previous = adapt_wrist_to_training_shape(previous)
            frames[camera] = current.copy()
            frames_prev[camera] = previous.copy()
        metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("schema") not in {
        "ramen_policy_observation_v1",
        "ramen_policy_observation_v2",
    }:
        raise ValueError(f"unsupported observation schema: {metadata.get('schema')!r}")
    if metadata.get("actuator_constructed") is not False:
        raise ValueError("observation dump does not attest actuator_constructed=false")
    return state, frames, frames_prev, metadata


def _run_dry_run(
    variant: str,
    config_path: Path,
    ticks: int,
    tick_hz: float,
    interface: str | None,
    no_real_state: bool,
    warmup_iters: int,
    state_dump_path: Path | None,
    observation_dump_path: Path | None,
) -> dict:
    entry, policy, dt_load = _load_policy(variant, config_path)
    vram_after_load = _vram_snapshot()
    policy.warmup(n_iter=warmup_iters)
    vram_after_warmup = _vram_snapshot()

    from inference.desktop.lower_policy.policies.base import Observation

    captured_state = None
    captured_frames = None
    captured_frames_prev = None
    captured_metadata = None
    if observation_dump_path is not None:
        (
            captured_state,
            captured_frames,
            captured_frames_prev,
            captured_metadata,
        ) = _load_captured_observation(observation_dump_path, policy)
        print(
            f"[smoke] captured 4-camera real observation loaded from "
            f"{observation_dump_path}; no DDS/actuator in model process",
            file=sys.stderr,
        )
    elif state_dump_path is not None:
        captured_state = _load_captured_state(state_dump_path, policy.STATE_DIM)
        print(
            f"[smoke] captured real state loaded from {state_dump_path} "
            f"(dim={captured_state.size}); no DDS/actuator in model process",
            file=sys.stderr,
        )

    src = None
    dex1_src = None
    fk = None
    if captured_state is None and not no_real_state:
        if interface is None:
            raise ValueError(
                "--interface or --state-dump required for real-state dry-run"
            )
        from inference.desktop.perception.g1_urdf_fk import (
            DEFAULT_URDF_PATH,
            G1WristFK,
        )

        urdf_path = Path(DEFAULT_URDF_PATH)
        if not urdf_path.exists():
            urdf_path = (Path(__file__).parents[4] / DEFAULT_URDF_PATH).resolve()
        fk = G1WristFK.from_urdf(str(urdf_path)) if urdf_path.exists() else None
        src, dex1_src = _get_joint_state(interface)
        print(
            "[smoke] JointStateSource + read-only Dex1StateSource init、warmup 1s...",
            file=sys.stderr,
        )
        time.sleep(1.0)

    # tick loop
    prev_joint_positions: np.ndarray | None = None
    prev_hand_state: np.ndarray | None = None
    prev_action_19d: np.ndarray | None = None
    frames_prev: dict | None = captured_frames_prev
    latencies_ms: list[float] = []
    chunks: list[np.ndarray] = []
    tick_ok_count = 0
    interval = 1.0 / max(tick_hz, 1.0)

    # variant 依存の skill_id / language (build_batch_dict の要求を満たすため)
    is_ramen_ori = entry.policy_type == "ramen_ori"
    skill_id = 0 if is_ramen_ori else None
    language = None if is_ramen_ori else "smoke dry-run"

    H, W = 480, 640

    for i in range(ticks):
        t_start = time.monotonic()
        if captured_state is not None:
            state = captured_state.copy()

        # joint_state 取得
        if captured_state is None and src is not None:
            snap = src.get()
            jp = (
                np.asarray(snap.position, dtype=np.float32)
                if snap is not None and snap.position is not None
                and len(snap.position) == 29
                else np.zeros(29, dtype=np.float32)
            )
        elif captured_state is None:
            jp = np.zeros(29, dtype=np.float32)
        if captured_state is None and dex1_src is not None:
            hand_snap = dex1_src.get()
            hand_position_rad = (
                hand_snap.position_rad
                if hand_snap is not None
                else np.zeros(2, dtype=np.float32)
            )
        elif captured_state is None:
            hand_position_rad = np.zeros(2, dtype=np.float32)

        if captured_state is None:
            from inference.desktop.lower_policy.policies.base import RawRobotState

            ee = (
                fk.compute_ee_state(jp)
                if fk is not None
                else np.zeros(12, dtype=np.float32)
            )
            raw = RawRobotState(
                joint_positions=jp,
                hand_state=hand_position_rad,
                ee_state=ee,
                last_action_19d=prev_action_19d,
                joint_positions_prev=prev_joint_positions,
                hand_state_prev=prev_hand_state,
            )
            state = policy.__class__.build_state_from_raw(raw)

        frames = (
            {cam: frame.copy() for cam, frame in captured_frames.items()}
            if captured_frames is not None
            else {cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in policy.CAMERAS}
        )
        obs = Observation(
            frames_bgr=frames,
            frames_bgr_prev=frames_prev,
            state=state,
            skill_id=skill_id,
            language=language,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )
        try:
            action = policy.predict(obs)
            chunks.append(action.action_chunk.copy())
            latencies_ms.append(action.latency_ms)
            tick_ok_count += 1
        except Exception as e:
            print(f"[tick {i}] predict failed: {type(e).__name__}: {e}", file=sys.stderr)

        # buffer update (次 tick が使う)
        if captured_state is None:
            prev_joint_positions = jp
            prev_hand_state = hand_position_rad.copy()
            if chunks:
                prev_action_19d = chunks[-1][0].astype(np.float32, copy=False)
        frames_prev = dict(frames)

        elapsed = time.monotonic() - t_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

    if src is not None:
        src.close()
    if dex1_src is not None:
        dex1_src.close()
    policy.close()

    if not chunks:
        return {"error": "no_successful_predict"}

    all_chunks = np.stack(chunks, axis=0)  # (T, chunk_len, action_dim)
    # chunk[0] の tick-to-tick 差分 (急変検出)
    if len(chunks) >= 2:
        first_step_series = all_chunks[:, 0, :]  # (T, action_dim)
        diff = np.abs(np.diff(first_step_series, axis=0))
        max_step_diff = float(diff.max())
        mean_step_diff = float(diff.mean())
    else:
        max_step_diff = None
        mean_step_diff = None

    latencies_sorted = sorted(latencies_ms)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else None

    # The 19-D actuator contract is waist(3), arms(14), Dex1(2).  Reporting
    # only a global min/max can hide an unsafe waist target behind otherwise
    # normal arm/hand values, so retain per-slice first-step diagnostics for
    # the non-actuating Phase 1.10/1.12 gate.
    first_step_series = all_chunks[:, 0, :]

    def _slice_diagnostics(start: int, stop: int) -> dict:
        values = first_step_series[:, start:stop]
        return {
            "min_per_dim": values.min(axis=0).astype(float).tolist(),
            "max_per_dim": values.max(axis=0).astype(float).tolist(),
            "mean_per_dim": values.mean(axis=0).astype(float).tolist(),
            "max_tick_delta_per_dim": (
                np.abs(np.diff(values, axis=0)).max(axis=0).astype(float).tolist()
                if values.shape[0] >= 2
                else [0.0] * values.shape[1]
            ),
        }

    result = {
        "mode": "dry-run",
        "variant": variant,
        "ticks_requested": ticks,
        "ticks_ok": tick_ok_count,
        "load_seconds": dt_load,
        "chunk_shape": list(all_chunks.shape[1:]),
        "chunk_finite_all": bool(np.all(np.isfinite(all_chunks))),
        "chunk_min": float(all_chunks.min()),
        "chunk_max": float(all_chunks.max()),
        "chunk_mean": float(all_chunks.mean()),
        "latency_ms": {
            "min": min(latencies_ms) if latencies_ms else None,
            "max": max(latencies_ms) if latencies_ms else None,
            "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
            "p95": p95,
        },
        "step_delta": {
            "max_first_step_diff": max_step_diff,
            "mean_first_step_diff": mean_step_diff,
        },
        "first_step_slices": {
            "waist_3d": _slice_diagnostics(0, 3),
            "arms_14d": _slice_diagnostics(3, 17),
            "dex1_2d": _slice_diagnostics(17, 19),
        },
        "real_state_used": captured_state is not None or not no_real_state,
        "state_source": (
            "captured_observation_npz" if captured_frames is not None
            else "captured_json" if captured_state is not None
            else "live_dds" if not no_real_state
            else "zeros"
        ),
        "camera_input": (
            {
                "real": True,
                "roles": [camera.value for camera in policy.CAMERAS],
                "camera_skew_ms": captured_metadata.get("camera_skew_ms"),
            }
            if captured_metadata is not None
            else {"real": False}
        ),
        "hand_state_unit": (
            "dex1_motor_output_rad"
            if captured_state is not None or not no_real_state
            else None
        ),
        "vram_after_load": vram_after_load,
        "vram_after_warmup": vram_after_warmup,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real ckpt load + dry-run forward smoke (Issue #125 Phase 1.9 / 1.10)."
    )
    parser.add_argument(
        "--variant", required=True,
        help="policy variant 名 (policy_config.yaml の policies section key)",
    )
    parser.add_argument(
        "--config", type=Path, default=_default_config_path(),
        help="policy_config.yaml path",
    )
    parser.add_argument(
        "--mode", required=True, choices=["load-only", "dry-run"],
        help="load-only=Phase 1.9 単発 load+warmup、dry-run=Phase 1.10 30-tick loop",
    )
    parser.add_argument("--ticks", type=int, default=30, help="dry-run tick 数")
    parser.add_argument("--tick-hz", type=float, default=30.0, help="dry-run tick rate")
    parser.add_argument(
        "--interface", type=str, default=None,
        help="cyclonedds NIC (dry-run で --no-real-state 未指定時に必須)",
    )
    parser.add_argument(
        "--no-real-state", action="store_true",
        help="dry-run で joint_state を zeros 固定 (JointStateSource 抜き、Model forward だけ verify)",
    )
    parser.add_argument(
        "--state-dump", type=Path, default=None,
        help="smoke_state_assembly JSONを使う（model envへDDS依存を混ぜない推奨経路）",
    )
    parser.add_argument(
        "--observation-dump", type=Path, default=None,
        help="capture_policy_observation NPZの実4カメラ＋stateを使う推奨経路",
    )
    parser.add_argument("--warmup-iters", type=int, default=3, help="warmup call 回数")
    parser.add_argument(
        "--dump-path", type=Path, default=None,
        help="結果 JSON dump path (省略時は stdout のみ)",
    )
    args = parser.parse_args()

    selected_inputs = sum(
        (
            bool(args.no_real_state),
            args.state_dump is not None,
            args.observation_dump is not None,
        )
    )
    if selected_inputs > 1:
        parser.error(
            "--no-real-state, --state-dump and --observation-dump are mutually exclusive"
        )

    if args.mode == "load-only":
        result = _run_load_only(args.variant, args.config, args.warmup_iters)
    else:
        result = _run_dry_run(
            args.variant, args.config, args.ticks, args.tick_hz,
            args.interface, args.no_real_state, args.warmup_iters,
            args.state_dump,
            args.observation_dump,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.dump_path is not None:
        args.dump_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[smoke] dumped to {args.dump_path}", file=sys.stderr)
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
