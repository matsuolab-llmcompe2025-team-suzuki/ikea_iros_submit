"""H-7 AWR twist: per-episode advantage-based sample weight (Issue #120 Alt-6)。

Design doc §8.5 の "本気" 差別化。**per-episode advantage で BC sample を重み付け**、
効率的な episode (短時間で完了) の trajectory 分布に policy を寄せる。純粋 AWR ではなく
"twist" (BC loss を advantage で weighting)、実装は sampler 側 multinomial 確率への
multiplier として実現 (loss reweight は Phase 2 拡張候補)。

# 現状 scope (Alt-6 v1)

**Duration-only signal**:
- Dataset B curated chunks (task 5+7、全 optimal verdict)
- Failure retry pattern 無し (curation で optimal のみ選抜済)
- ✅ `episode.length` (frame 数) から duration advantage 計算

**Failure-retry hybrid は Alt-6 v2 (別 Alt)** で Dataset A 対応時に追加。

# 統合先

`CurriculumSampler.__init__(..., advantage_weights=...)` に inject、compute_stage_weights
後に要素毎積で multinomial 確率を bias。curriculum enabled/disabled にかかわらず統合可能。
"""

from __future__ import annotations

import numpy as np


def compute_duration_advantage(
    episode_lengths: np.ndarray,
    method: str = "inverse",
    temperature: float = 1.0,
) -> np.ndarray:
    """Duration → per-episode advantage weight (mean=1 に normalize)。

    Args:
        episode_lengths: (num_episodes,) 各 ep の frame 数、>0 必須
        method: advantage 計算方式
            - "inverse": w_ep = mean(L) / L_ep  (短 ep = 高 weight、直感的)
            - "z_score": w_ep = exp(-(L - mean)/std/temperature) → mean=1 rescale
            - "rank":    順位ベース、極端値の影響回避
        temperature: z_score のみ使用 (>1 で uniform 寄り、<1 で bias 強)

    Returns:
        (num_episodes,) float64 weight、mean(weights)=1

    Raises:
        ValueError: episode_lengths <= 0、method 未知、temperature<=0
    """
    L = np.asarray(episode_lengths, dtype=np.float64)
    if L.ndim != 1:
        raise ValueError(f"episode_lengths must be 1-D, got shape {L.shape}")
    if (L <= 0).any():
        raise ValueError(f"episode_lengths must be > 0, got min={L.min()}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    if method == "inverse":
        w = L.mean() / L
    elif method == "z_score":
        mu, sd = L.mean(), L.std()
        if sd == 0:
            w = np.ones_like(L)
        else:
            # 短 ep (L < mu) が高 weight、標準化してから exp
            w = np.exp(-(L - mu) / sd / temperature)
    elif method == "rank":
        # 短い順に高 rank、rank/N を weight (0..1)、その後 mean=1 rescale
        order = np.argsort(L)  # 短い順 index
        rank = np.empty_like(L)
        rank[order] = np.arange(len(L))  # 短い方が rank 0
        w = (len(L) - rank) / len(L)     # 短い→高、範囲 (0, 1]
    else:
        raise ValueError(
            f"unknown method {method!r} (expected 'inverse'|'z_score'|'rank')"
        )

    # mean=1 に normalize
    mean_w = w.mean()
    if mean_w == 0:
        raise ValueError("advantage weight mean=0, cannot normalize")
    return w / mean_w


def broadcast_to_frames(
    ep_weights: np.ndarray,
    episode_lengths: np.ndarray,
) -> np.ndarray:
    """per-episode weight を per-frame weight に expand。

    frame i は所属する ep の weight を継承。ep 内全 frame が同 weight。

    Args:
        ep_weights: (num_episodes,) advantage weight
        episode_lengths: (num_episodes,) 各 ep の frame 数

    Returns:
        (sum(episode_lengths),) per-frame weight
    """
    if len(ep_weights) != len(episode_lengths):
        raise ValueError(
            f"ep_weights ({len(ep_weights)}) vs episode_lengths ({len(episode_lengths)}) mismatch"
        )
    return np.repeat(ep_weights, episode_lengths.astype(np.int64))
