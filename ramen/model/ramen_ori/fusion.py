"""Fusion transformer variants (RAMEN-Ori、Issue #115 で G-1、Issue #120 Alt-1 で G-2/G-3 追加)。

3 variant (design doc §4.3 G axis):

- **G-1 `FusionTransformer`** (Phase 1 baseline、最軽量): 全 token concat → self-attn
  6 layer で mix。context 出力 shape = 入力 tokens 同数。
- **G-2 `FusionInterleaved`** (Phase 2 bench 対抗、Pi0.5 style): 奇偶 layer で
  self-attn / cross-attn を交互。learnable query set (Q) が全 token を attend、
  self-attn は query 間相互。context 出力 shape = num_queries。
- **G-3 `FusionCrossAttn`** (design doc Phase 1 default、Perceiver / SmolVLA 系):
  cross-attn のみ、learnable query が全 token を attend。self-attn 無し。context
  出力 shape = num_queries。

# Shape (共通)

- input:
    - tokens:         (B, N_total, d_model)、model.py で 各 path から concat 済
    - attention_mask: (B, N_total) bool、True=attend、False=無視 (OBB padded 用)、Optional
- output: (context, context_mask)
    - context:        (B, M, d_model)、Action Expert が cross-attn で参照
      * G-1: M = N_total (input tokens 数)
      * G-2/G-3: M = num_queries
    - context_mask:   (B, M) bool、Action Expert 側 cross-attn の key_padding_mask 用
      * G-1: input mask をそのまま pass-through
      * G-2/G-3: 全 True (queries は全 valid)

# Design 共通

- pre-LN、GELU、batch_first=True、norm_first=True
- 最終 LN を post 段に 1 個
- RoPE は Phase 1 skip
- Modality embedding は Phase 1 skip、token 分布の差で自然識別想定
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ramen_ori.attention import qk_norm_attention


class _QKNormEncoderLayer(nn.Module):
    """`nn.TransformerEncoderLayer` (pre-LN、GELU、dropout 0.1) と同じ計算・同じ param 名に、
    q/k の head ごとの RMSNorm を足した層 (Issue #141)。"""

    def __init__(self, d_model: int, num_heads: int, mlp_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.q_norm = nn.RMSNorm(d_model // num_heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(d_model // num_heads, eps=1e-6)
        self.linear1 = nn.Linear(d_model, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout1(qk_norm_attention(self.self_attn, self.q_norm, self.k_norm, h, h, key_padding_mask))
        return x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))


class _QKNormEncoder(nn.Module):
    """`_QKNormEncoderLayer` を重ねたもの。`nn.TransformerEncoder` と同じ `layers` と呼び出し方。"""

    def __init__(self, d_model: int, num_heads: int, mlp_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_QKNormEncoderLayer(d_model, num_heads, mlp_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x


class FusionTransformer(nn.Module):
    """G-1 Standard self-attn Fusion (Phase 1 baseline、最軽量)。

    全 token を concat → self-attn 6 layer。output shape は input と同数 tokens。

    qk_norm=True なら attention の q/k を head ごとに RMSNorm する (Issue #141、`attention.py`)。
    False (既定) は `nn.TransformerEncoder` のままで、Phase K の ckpt と同じ構造。
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        if qk_norm:
            self.encoder = _QKNormEncoder(d_model, num_heads, mlp_dim, num_layers)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # enable_nested_tensor=False: pre-LN と nested tensor optimization は非互換
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=False
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # PyTorch src_key_padding_mask: True = ignore、本 module の attention_mask は
        # True = attend 慣例、反転して渡す
        kpm = ~attention_mask if attention_mask is not None else None
        out = self.encoder(tokens, src_key_padding_mask=kpm)
        return self.norm(out), attention_mask


class _CrossAttnBlock(nn.Module):
    """1 layer of cross-attn (query attends context tokens) + MLP。pre-LN + residual。"""

    def __init__(self, d_model: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # pre-LN cross-attn
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=context_key_padding_mask, need_weights=False
        )
        query = query + attn_out
        # pre-LN FFN
        query = query + self.ffn(self.norm_ffn(query))
        return query


class _SelfAttnBlock(nn.Module):
    """1 layer of self-attn + MLP。pre-LN + residual。query の相互 attend 用。"""

    def __init__(self, d_model: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm_attn(x)
        attn_out, _ = self.self_attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FusionCrossAttn(nn.Module):
    """G-3 Cross-attn only Fusion (design doc Phase 1 default、Perceiver / SmolVLA 系)。

    Learnable query set (num_queries 個) が cross-attn で context tokens を attend。
    self-attn は無し、最も軽量。output shape = num_queries。
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        num_queries: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        # learnable query embed (init std=0.02、trunc_normal)
        self.queries = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.layers = nn.ModuleList(
            [_CrossAttnBlock(d_model, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = tokens.shape[0]
        query = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, Q, d)
        # cross-attn の key_padding_mask: True = ignore
        kpm = ~attention_mask if attention_mask is not None else None
        for layer in self.layers:
            query = layer(query, tokens, kpm)
        out = self.norm(query)
        # output mask: queries は全 valid
        out_mask = torch.ones(B, self.num_queries, dtype=torch.bool, device=tokens.device)
        return out, out_mask


class FusionInterleaved(nn.Module):
    """G-2 Interleaved self-cross-attn Fusion (Phase 2 bench 対抗、Pi0.5 style)。

    Layer 0/2/4/...: cross-attn (query が context を attend)
    Layer 1/3/5/...: self-attn (query 間相互)
    最初 cross で情報引き込み → self で query 内 mix、を交互。
    output shape = num_queries。
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        num_queries: int = 32,
    ) -> None:
        super().__init__()
        if num_layers % 2 != 0:
            raise ValueError(
                f"FusionInterleaved needs even num_layers (cross/self ペア)、got {num_layers}"
            )
        self.d_model = d_model
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # 偶数 index = cross、奇数 index = self、ペアで num_layers/2 個ずつ
        layers = []
        for i in range(num_layers):
            if i % 2 == 0:
                layers.append(_CrossAttnBlock(d_model, num_heads, mlp_dim))
            else:
                layers.append(_SelfAttnBlock(d_model, num_heads, mlp_dim))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = tokens.shape[0]
        query = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        kpm = ~attention_mask if attention_mask is not None else None
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                query = layer(query, tokens, kpm)  # cross-attn
            else:
                query = layer(query, None)         # self-attn (query 相互、pad 無し)
        out = self.norm(query)
        out_mask = torch.ones(B, self.num_queries, dtype=torch.bool, device=tokens.device)
        return out, out_mask
