"""Precompute teacher Δq stats for relative action space (Issue #129 Phase F、2026-08-31)。

Run 3/5 の relative action space で arms 14 dim を Δq に normalize する時の per-dim
mean/std を dataset 全 frame から計算し、torch.save で保存する。data_lerobot と model
の両方が load して同一 stats を使う (bench 純度)。

# 使い方

    pixi run python -m model.ramen_ori.scripts.compute_relative_action_stats \
        --data-config data=real_task5_7 \
        --output outputs/relative_stats_task5_7.pt \
        [--num-workers 4]

出力: torch.save された dict {"mean": (14,) float32, "std": (14,) float32}。
dataset 名別に per-recipe stats を保存する慣行 (rotate+move / +insert 等で mix 変わる)。

# 計算方針

- Δq[0] = teacher_arms[0] - state_arms_current (per frame)
- Δq[k>0] = teacher_arms[k] - teacher_arms[k-1] (chunk 内)
- 全 frame × 全 chunk step の Δq を collect → per-dim mean/std
- Per-dim = 14 dim (left arm 7 + right arm 7)、hand は Δq 化しないので stats 不要

# Cost 見積 (6.27M frames 想定、task 5+7)

- 1 frame あたり ~10 ms (LeRobot decode + numpy) → 6.27M × 10ms = 17 h single thread
- num_workers=4 で ~4-5 h、num_workers=8 で ~2-3 h
- CPU bound (GPU 不要)、Sakura 停止中に走らせても OK
- 実際は "全 frame" である必要はなく、10-20% subsample で十分精度出る (mean/std 収束)

# Subsample オプション

`--subsample-ratio 0.1` で 10% ランダム抽出 (default 1.0 = 全 frame)、smoke / debug 用。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from model.ramen_ori.state_derive import (
    ARMS_HAND_SOURCE_INDEX_MAP,
    STATE71_ARMS_SLICE,
    derive_state_71d,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _iter_dataset_dq_arms(ds, subsample_ratio: float, seed: int):
    """Dataset を iterate して per-frame の (chunk_len, 14) Δq をひとつずつ yield。

    実装は data_lerobot._transform_item の relative branch と同一計算を再現。
    subsample_ratio < 1.0 なら決定的にランダム skip (seed 固定)。
    """
    from model.ramen_ori.data_lerobot import _to_numpy, _scalar_from  # noqa: F401 (helper 共有)

    rng = np.random.default_rng(seed)
    N = len(ds)
    if subsample_ratio < 1.0:
        keep_mask = rng.random(N) < subsample_ratio
        keep_indices = np.where(keep_mask)[0]
    else:
        keep_indices = np.arange(N)
    log.info(f"iterating {len(keep_indices)}/{N} frames (subsample_ratio={subsample_ratio})")

    for i_idx, idx in enumerate(keep_indices):
        if i_idx % 1000 == 0:
            log.info(f"  progress: {i_idx}/{len(keep_indices)}")
        item = ds._base[int(idx)]
        # data_lerobot と同じ流れ (compact 版)
        q_current = np.asarray(item["observation.state.robot_q_current"])   # (2, 36)
        hand_state = np.asarray(item["observation.state.hand_state"])       # (2, 2)
        ee_state = np.asarray(item["observation.state.ee_state"])           # (1, 12)
        # action は row 0 = 1 frame 前の指令 (Issue #141 RO-3)、row 1.. が正解 chunk
        q_desired = np.asarray(item["action.robot_q_desired"])[1:]          # (chunk_len, 36)
        hand_cmd = np.asarray(item["action.hand_cmd"])[1:]                  # (chunk_len, 2)
        state_current_38 = np.concatenate([q_current[1], hand_state[1]]).astype(np.float32)
        ee_state_12 = ee_state[0].astype(np.float32)
        state_71 = derive_state_71d(
            state_current=state_current_38,
            action_prev=None,  # tracking_err / velocity は Δq stats に無関係、None で OK
            ee_state=ee_state_12,
            state_prev=None,
        )
        # arms current (state[3:17])
        arms_current_14 = state_71[STATE71_ARMS_SLICE].astype(np.float32)
        # teacher arms 14 chunk (arms_hand[..:14])
        action_full = np.concatenate([q_desired, hand_cmd], axis=1).astype(np.float32)
        arms_hand_idx = np.asarray(ARMS_HAND_SOURCE_INDEX_MAP)
        action_16 = action_full[:, arms_hand_idx]
        arms_teacher_14 = action_16[:, 0:14]  # (chunk_len, 14)
        # Δq: [0] = teacher[0] - current、[k>0] = teacher[k] - teacher[k-1]
        dq_0 = (arms_teacher_14[0] - arms_current_14).reshape(1, 14)
        dq_rest = arms_teacher_14[1:] - arms_teacher_14[:-1]
        dq = np.concatenate([dq_0, dq_rest], axis=0).astype(np.float32)  # (chunk_len, 14)
        yield dq


def compute_stats(ds, subsample_ratio: float = 1.0, seed: int = 42) -> dict[str, np.ndarray]:
    """Dataset 全 frame (or subsample) から (mean, std) を per-dim (14,) で計算。"""
    all_dq: list[np.ndarray] = []
    for dq in _iter_dataset_dq_arms(ds, subsample_ratio, seed):
        all_dq.append(dq)
    stacked = np.concatenate(all_dq, axis=0)   # (N_total_dq, 14)
    log.info(f"total Δq samples: {stacked.shape[0]}, dim: {stacked.shape[1]}")
    mean = stacked.mean(axis=0).astype(np.float32)  # (14,)
    std = stacked.std(axis=0).astype(np.float32)    # (14,)
    log.info(f"mean per dim: {mean}")
    log.info(f"std per dim:  {std}")
    return {"mean": mean, "std": std}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config",
        required=True,
        help="Hydra data group name (e.g. 'real_task5_7')。configs/data/{name}.yaml を load",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output stats file (torch.save)")
    parser.add_argument(
        "--subsample-ratio",
        type=float,
        default=1.0,
        help="0.1 で 10% subsample (debug / smoke)、default 1.0 = 全 frame",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="base", overrides=[f"data={args.data_config}"])

    from hydra.utils import instantiate

    log.info(f"instantiating dataset: {cfg.data._target_}")
    ds = instantiate(cfg.data)

    stats = compute_stats(ds, subsample_ratio=args.subsample_ratio, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"mean": torch.from_numpy(stats["mean"]), "std": torch.from_numpy(stats["std"])},
        args.output,
    )
    log.info(f"saved to {args.output}")


if __name__ == "__main__":
    main()
