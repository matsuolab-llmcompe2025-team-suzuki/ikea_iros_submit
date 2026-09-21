"""QK-norm の attention (Issue #141)。

`nn.MultiheadAttention` の重み (in_proj・out_proj) をそのまま使い、q と k を head ごとに正規化してから
scaled dot product attention を計算する。q・k の重みが学習で大きくなっても、logit は √head_dim × (norm の scale)²
程度に収まる。c32 の 1 回目・2 回目 (clip あり) では fusion の 0 層目の最大 logit が 5k 約 900 → 30k 約 15 万と増え続け、
vision の token の attention の約 6 割が state の token に張り付いて、勾配が state の入口の層に集まった。

param 名は `nn.MultiheadAttention` のまま (q/k の norm は呼ぶ側の module が持つ)。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def qk_norm_attention(
    mha: nn.MultiheadAttention,
    q_norm: Callable[[torch.Tensor], torch.Tensor],
    k_norm: Callable[[torch.Tensor], torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """query (B, Lq, d) が key (B, Lk, d) を attend した結果 (B, Lq, d) を返す (key と value は同じ入力)。

    mha は batch_first で q・k・v の次元が同じもの。q_norm / k_norm は head の次元 (d // num_heads) にかける。
    key_padding_mask (B, Lk) は nn.MultiheadAttention と同じ向き (True = 無視)。attention の dropout は mha の設定。
    """
    d, h = mha.embed_dim, mha.num_heads
    q = F.linear(query, mha.in_proj_weight[:d], mha.in_proj_bias[:d])
    k, v = F.linear(key, mha.in_proj_weight[d:], mha.in_proj_bias[d:]).chunk(2, dim=-1)
    q = q_norm(q.unflatten(-1, (h, d // h))).transpose(1, 2)  # (B, h, Lq, head_dim)
    k = k_norm(k.unflatten(-1, (h, d // h))).transpose(1, 2)
    v = v.unflatten(-1, (h, d // h)).transpose(1, 2)
    # SDPA の bool mask は True = attend
    attn_mask = None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=mha.dropout if mha.training else 0.0
    )
    return mha.out_proj(out.transpose(1, 2).flatten(2))
