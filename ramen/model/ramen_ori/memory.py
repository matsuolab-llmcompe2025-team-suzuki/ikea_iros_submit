"""memory の token (Issue #141 Phase 7、変種「memory の入力」)。

`memory_features.MemoryTracker` が作る 51 個 → 学習の範囲 (p1〜p99) で切り詰め → 平均・std で正規化 →
小さな MLP → token 1 個 (Fusion に state の token と同じ入れ方で足す)。
切り詰めの範囲と統計は学習の開始時に計算して buffer に入れる (ckpt に保存され、推論も同じ値を使う)。

学習時 (train mode) は、自分の動きの 18 個をまとめて `motion_dropout` の確率で 0 (= 正規化後の平均) にし、
隠した印を 1 個足す (直近の動きを続ける癖 = copycat → drift を防ぐ。速度の平均はほぼ 0 なので、印で「動いていない」と区別する)。
eval mode (val と推論) では隠さず、印は 0。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.ramen_ori.memory_features import MEMORY_DIM, MOTION_SLICE
from model.ramen_ori.normalization import Normalizer


class MemoryEncoder(nn.Module):
    def __init__(
        self,
        memory_dim: int = MEMORY_DIM,
        d_model: int = 512,
        hidden_dim: int | None = None,
        motion_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if memory_dim != MEMORY_DIM:
            raise ValueError(f"memory_dim must be {MEMORY_DIM} (memory_features.MEMORY_LAYOUT), got {memory_dim}")
        if not 0.0 <= motion_dropout <= 1.0:
            raise ValueError(f"motion_dropout must be in [0, 1], got {motion_dropout}")
        hidden_dim = hidden_dim or d_model
        self.memory_dim = memory_dim
        self.d_model = d_model
        self.motion_dropout = float(motion_dropout)
        self.num_flags = 1 if self.motion_dropout > 0 else 0

        self.register_buffer("clip_low", torch.zeros(memory_dim))
        self.register_buffer("clip_high", torch.zeros(memory_dim))
        self.normalizer = Normalizer(memory_dim)
        self.fc1 = nn.Linear(memory_dim + self.num_flags, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def set_stats(self, clip_low, clip_high, mean, std) -> None:
        """学習の開始時に計算した切り詰めの範囲と、切り詰めた後の平均・std を入れる。"""
        low = torch.as_tensor(clip_low, dtype=torch.float32)
        high = torch.as_tensor(clip_high, dtype=torch.float32)
        if low.shape != (self.memory_dim,) or high.shape != (self.memory_dim,) or bool((low > high).any()):
            raise ValueError(f"clip の範囲が不正: low {tuple(low.shape)} / high {tuple(high.shape)}")
        self.clip_low.copy_(low.to(self.clip_low.device))
        self.clip_high.copy_(high.to(self.clip_high.device))
        self.normalizer.set_stats(mean, std)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """memory (B, 51) → (B, 1, d_model)。"""
        x = self.normalizer.normalize(torch.clamp(memory.float(), self.clip_low, self.clip_high))
        if self.num_flags:
            B = x.shape[0]
            if self.training:
                hide = torch.rand(B, 1, device=x.device) < self.motion_dropout
                x = x.clone()
                x[:, MOTION_SLICE] = x[:, MOTION_SLICE].masked_fill(hide, 0.0)
                flag = hide.to(x.dtype)
            else:
                flag = x.new_zeros(B, 1)
            x = torch.cat([x, flag], dim=-1)
        return self.fc2(self.act(self.fc1(x))).unsqueeze(1)
