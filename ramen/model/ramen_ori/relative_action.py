"""Relative action space helper (Issue #129 Phase F、2026-08-31)。

Run 3 / Run 5 用の action space refactor: arms 14 dim を Δq (relative) 化、hand 2 dim
は absolute pass-through。Flow Matching の signal-to-noise 対策で arms Δq は
unit-variance normalize (mean/std は事前 precompute、`compute_relative_action_stats.py`)。

# 何故 hand は absolute のままか (Session 32 論点 M2)

Hand は discrete state (open / close 前後)、Δq 化すると signal 潰れる。GR00T の
`relative_exclude_joints=[hand, waist, base_height, navigate]` と同じ思想。

# 何故 arms Δq を normalize するか

Flow Matching は x_0 ~ N(0, 1) noise、x_1 = target action で v = x_1 - x_0 を予測する。
Δq (0.01 rad 級) は noise variance 1 に埋もれる (signal-to-noise 極低) → 学習困難。
unit variance normalize で SNR を absolute (~0.5-1.0 rad 級) と同水準に。

# API

    from model.ramen_ori.relative_action import (
        load_relative_stats,
        compute_teacher_arms_dq,
        normalize_arms_dq,
        denormalize_arms_dq,
        reconstruct_arms_abs_from_dq_norm,
    )

    stats = load_relative_stats("outputs/relative_stats_task5_7.pt")   # {"mean": (14,), "std": (14,)}
    dq = compute_teacher_arms_dq(arms_teacher_chunk, arms_current)      # (..., chunk, 14)
    dq_norm = normalize_arms_dq(dq, stats["mean"], stats["std"])
    # inference / L4 で復元:
    arms_pred_abs = reconstruct_arms_abs_from_dq_norm(
        dq_norm=pred_dq_norm, arms_current=arms_current,
        mean=stats["mean"], std=stats["std"],
    )   # (..., chunk, 14) absolute arms trajectory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# stats file I/O
# ---------------------------------------------------------------------------


def load_relative_stats(path: str | Path) -> dict[str, torch.Tensor]:
    """Precompute した stats file (torch.save 済 dict) を load。

    Expected keys: "mean" (14,) float、"std" (14,) float。std==0 の dim は 1 で clip して
    normalize 側で無効化 (variance 0 = 学習で動かない dim だが Δq=0 で safe fallback)。
    """
    stats: dict[str, Any] = torch.load(path, weights_only=False, map_location="cpu")
    if "mean" not in stats or "std" not in stats:
        raise ValueError(
            f"relative_stats file {path} must have 'mean' and 'std' keys, got {list(stats.keys())}"
        )
    mean = stats["mean"]
    std = stats["std"]
    if not isinstance(mean, torch.Tensor):
        mean = torch.as_tensor(mean, dtype=torch.float32)
    if not isinstance(std, torch.Tensor):
        std = torch.as_tensor(std, dtype=torch.float32)
    if mean.shape != (14,) or std.shape != (14,):
        raise ValueError(
            f"relative_stats mean/std must be (14,), got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
        )
    return {"mean": mean.float(), "std": std.float()}


# ---------------------------------------------------------------------------
# Δq computation + normalize / denormalize
# ---------------------------------------------------------------------------


def compute_teacher_arms_dq(
    arms_teacher_chunk: torch.Tensor,
    arms_current: torch.Tensor,
) -> torch.Tensor:
    """Teacher arms trajectory を Δq に変換。

    - chunk step 0 の Δq = teacher[0] - current  (student は現状から teacher [0] へ ramp up)
    - chunk step k>0 の Δq = teacher[k] - teacher[k-1]

    Args:
        arms_teacher_chunk: (..., chunk_len, 14) teacher absolute arms trajectory
        arms_current:       (..., 14) 現 frame の arms current q (batch state から抽出)

    Returns:
        (..., chunk_len, 14) Δq tensor (raw scale、normalize 前)
    """
    if arms_teacher_chunk.shape[-1] != 14:
        raise ValueError(
            f"arms_teacher_chunk last dim must be 14, got {arms_teacher_chunk.shape}"
        )
    if arms_current.shape[-1] != 14:
        raise ValueError(
            f"arms_current last dim must be 14, got {arms_current.shape}"
        )
    # dq[0] = teacher[0] - current
    dq_0 = arms_teacher_chunk[..., 0:1, :] - arms_current.unsqueeze(-2)  # (..., 1, 14)
    # dq[k>0] = teacher[k] - teacher[k-1]
    dq_rest = arms_teacher_chunk[..., 1:, :] - arms_teacher_chunk[..., :-1, :]  # (..., chunk-1, 14)
    return torch.cat([dq_0, dq_rest], dim=-2)  # (..., chunk_len, 14)


def normalize_arms_dq(
    dq: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Δq を per-dim unit variance に normalize: (dq - mean) / max(std, eps)。"""
    if dq.shape[-1] != 14:
        raise ValueError(f"dq last dim must be 14, got {dq.shape}")
    return (dq - mean) / std.clamp(min=eps)


def denormalize_arms_dq(
    dq_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """normalize の逆: dq = dq_norm * std + mean。"""
    if dq_norm.shape[-1] != 14:
        raise ValueError(f"dq_norm last dim must be 14, got {dq_norm.shape}")
    return dq_norm * std + mean


def reconstruct_arms_abs_from_dq_norm(
    dq_norm: torch.Tensor,
    arms_current: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """normalized Δq → arms absolute trajectory の復元 (inference / L4 rel で使用)。

    absolute_arms[k] = arms_current + cumsum(denorm_dq[0..k])
                     = arms_current + sum_{i=0}^{k} (dq_norm[i] * std + mean)

    chunk step 0 は arms_current + denorm_dq[0] (= teacher[0] with perfect prediction)。

    Args:
        dq_norm:      (..., chunk_len, 14) normalized Δq
        arms_current: (..., 14) 現 frame arms current q
        mean, std:    (14,) precompute stats

    Returns:
        (..., chunk_len, 14) absolute arms trajectory
    """
    dq_denorm = denormalize_arms_dq(dq_norm, mean, std)  # (..., chunk_len, 14)
    return arms_current.unsqueeze(-2) + torch.cumsum(dq_denorm, dim=-2)
