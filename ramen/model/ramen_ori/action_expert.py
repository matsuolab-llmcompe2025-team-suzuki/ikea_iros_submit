"""B-7 Flow Matching DiT action expert module (RAMEN-Ori、Issue #115)。

Fusion からの context を条件に、chunk=16 step × action_dim=19 の action chunk を
Flow Matching で予測。DiT (Peebles 2023) の AdaLN-Zero + Pi0.5 style の
cross-attention conditioning を踏襲。

# 責務分離

- `FlowMatchingDiT` (pure nn.Module): `.forward(x_noised, t, context, context_mask)` の
  shape 変換だけを持つ、network eval のみ (weights の勾配計算に使う)
- `flow_matching_loss(model, target, context, ...)`: training 用 L2 loss
- `sample_action(model, context, n_steps, ...)`: inference 用 Euler 積分

# Shape

- `FlowMatchingDiT.forward(x, t, context, context_mask)`:
    - x:            (B, chunk_len, action_dim)  — noised action
    - t:            (B,)                        — flow matching time ∈ [0, 1]
    - context:      (B, N_context, context_dim) — Fusion 出力
    - context_mask: (B, N_context) bool         — True = attend、False = 無視
    - returns:      (B, chunk_len, action_dim)  — velocity prediction

# DiT config default

embed=768, depth=12, heads=12, mlp_ratio=4 (canonical DiT-B/2 相当)。
per-block AdaLN modulation (Linear(dim, 9*dim)) が予想以上に param 食うので、
実 param は ~170-180M (design doc "~150M" 記述に対して +20% 程度)。
Phase 2 で depth 10 に落とすと 150M に近づく。

# AdaLN-Zero pattern (DiT paper)

- 各 block の adaln[-1] Linear を zero-init → 初期 modulation = 0 → block は identity 相当
- output_proj も zero-init → v_pred = 0 at init → 初期 Euler 積分は x_0 (noise) が
  ほぼそのまま x_1 に届く安定化 pattern
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.ramen_ori.attention import qk_norm_attention


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, row_mask: torch.Tensor | None
) -> torch.Tensor:
    """(B, T, D) の MSE を、row_mask (B, T) が True の行だけで平均する (Issue #141 RO-2)。

    row_mask=None なら全要素の平均 (F.mse_loss と同じ)。区間末尾の埋め草 (最終 frame の繰り返し)
    の行を False にして loss から外す。各 sample の行 0 は常に埋め草でない。
    """
    se = (pred - target) ** 2
    if row_mask is None:
        return se.mean()
    m = row_mask.to(se.dtype).unsqueeze(-1)
    return (se * m).sum() / (m.sum() * se.shape[-1])


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DiT AdaLN modulation: (1 + scale) * x + shift、token 軸に broadcast。"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimeEmbedding(nn.Module):
    """Sinusoidal + 2-layer MLP time embedding。"""

    def __init__(self, dim: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, freq_dim)
        return self.mlp(emb)


class DiTBlock(nn.Module):
    """DiT block: AdaLN-Zero + self-attn + cross-attn (to context) + MLP。

    qk_norm=True なら self-attn と cross-attn の q/k を head ごとに RMSNorm する (Issue #141、`attention.py`)。
    False (既定) は nn.MultiheadAttention のままで、Phase K の ckpt と同じ構造。
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qk_norm: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.qk_norm = qk_norm
        if qk_norm:
            head_dim = dim // num_heads
            self.attn_q_norm = nn.RMSNorm(head_dim, eps=1e-6)
            self.attn_k_norm = nn.RMSNorm(head_dim, eps=1e-6)
            self.cross_q_norm = nn.RMSNorm(head_dim, eps=1e-6)
            self.cross_k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        # AdaLN-Zero: 9 modulation params per block (shift, scale, gate) × 3 (self, cross, mlp)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 9),
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mods = self.adaln(t_embed).chunk(9, dim=-1)
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = mods

        h = _modulate(self.norm1(x), shift1, scale1)
        if self.qk_norm:
            h = qk_norm_attention(self.attn, self.attn_q_norm, self.attn_k_norm, h, h)
        else:
            h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * h

        h = _modulate(self.norm2(x), shift2, scale2)
        if self.qk_norm:
            h = qk_norm_attention(
                self.cross_attn, self.cross_q_norm, self.cross_k_norm, h, context, context_key_padding_mask
            )
        else:
            h, _ = self.cross_attn(
                h, context, context,
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )
        x = x + gate2.unsqueeze(1) * h

        h = _modulate(self.norm3(x), shift3, scale3)
        h = self.mlp(h)
        x = x + gate3.unsqueeze(1) * h

        return x


class FlowMatchingDiT(nn.Module):
    def __init__(
        self,
        chunk_len: int = 16,
        action_dim: int = 19,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        context_dim: int = 512,
        freq_dim: int = 256,
        qk_norm: bool = False,  # Issue #141: attention の q/k を head ごとに RMSNorm (DiTBlock)
    ) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.context_dim = context_dim

        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, chunk_len, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.time_embed = TimeEmbedding(embed_dim, freq_dim=freq_dim)

        # Fusion 出力 (d_model=512) と DiT 内部 embed (768) の dim mismatch を吸収
        self.context_proj = nn.Linear(context_dim, embed_dim)

        self.blocks = nn.ModuleList(
            [DiTBlock(embed_dim, num_heads, mlp_ratio, qk_norm=qk_norm) for _ in range(depth)]
        )

        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        # 最終 AdaLN: shift + scale (gate 無し)
        self.adaln_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 2),
        )
        nn.init.zeros_(self.adaln_final[-1].weight)
        nn.init.zeros_(self.adaln_final[-1].bias)

        self.output_proj = nn.Linear(embed_dim, action_dim)
        # zero-init: v_pred = 0 at init、Euler 積分の初手が identity 相当 = 学習安定化
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.action_proj(x) + self.pos_embed  # (B, chunk_len, embed_dim)
        t_e = self.time_embed(t)  # (B, embed_dim)
        ctx = self.context_proj(context)  # (B, N_context, embed_dim)

        # PyTorch MultiheadAttention の key_padding_mask: True = ignore
        # 本 module の context_mask 慣例: True = attend、なので反転
        kpm = ~context_mask if context_mask is not None else None

        for block in self.blocks:
            h = block(h, t_e, ctx, kpm)

        shift, scale = self.adaln_final(t_e).chunk(2, dim=-1)
        h = _modulate(self.norm_final(h), shift, scale)
        return self.output_proj(h)

    # method-based API (Issue #120 Alt-7 で追加、model.py がこれを呼ぶ)
    def compute_loss(
        self,
        target_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        row_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return flow_matching_loss(self, target_action, context, context_mask, row_mask)

    def compute_loss_with_prediction(
        self,
        target_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        row_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Issue #129 Phase B (2026-08-31): FK の loss 用の 1-step Euler
        end prediction を loss と同時に返す (BC forward 1 回を再利用、compute 追加なし)。

        Returns:
            (loss, x_1_hat):
              loss  = MSE(v_pred, v_target)  (BC と同じ、row_mask の行だけ)
              x_1_hat = x_t + v_pred * (1 - t)  1-step Euler で x_1 相当を復元
                        (v_pred = v_target なら x_1_hat = x_1、Flow Matching の identity)
        """
        return flow_matching_loss_with_prediction(
            self, target_action, context, context_mask, row_mask
        )

    # RTC guidance (prefix / velocity_strength) に対応しているか。MeanFlow は
    # 既定 1 step で ramp を刻む余地が無いため非対応。
    SUPPORTS_RTC_GUIDANCE: bool = True

    def sample(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        n_steps: int = 6,
        *,
        prefix: torch.Tensor | None = None,
        velocity_strength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return sample_action(
            self,
            context,
            context_mask,
            n_steps=n_steps,
            prefix=prefix,
            velocity_strength=velocity_strength,
        )


def flow_matching_loss(
    model: FlowMatchingDiT,
    target_action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear interpolation + L2 velocity loss。

    - x_0 ~ N(0, 1)、x_1 = target_action、x_t = (1-t)*x_0 + t*x_1
    - v_target = x_1 - x_0
    - v_pred = model(x_t, t, context, context_mask)
    - loss = MSE(v_pred, v_target)、row_mask (B, T) が True の行だけで平均 (区間末尾の埋め草を外す)
    - t は uniform ∈ [0, 1] sampling (Phase 2 で logit-normal / OT 検討)
    """
    loss, _ = flow_matching_loss_with_prediction(
        model, target_action, context, context_mask, row_mask
    )
    return loss


def flow_matching_loss_with_prediction(
    model: FlowMatchingDiT,
    target_action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow Matching loss + 1-step Euler end prediction (Issue #129 Phase B、2026-08-31)。

    L4 FK anchor loss で使う: pred の 1-step Euler 復元 x_1_hat を FK に食わせて
    teacher wrist 3D 位置と MSE を取る。BC loss (v_pred vs v_target) と同一 forward の
    v_pred を再利用するので compute 追加なし。

    x_1_hat identity: v_pred = v_target なら x_1_hat = x_t + (x_1 - x_0)(1-t) = x_1、
    つまり L4 = MSE(FK(x_1_hat), FK(x_1)) は v_pred 誤差の物理空間投影。

    Returns:
        (loss, x_1_hat):
          loss   = MSE(v_pred, v_target)
          x_1_hat = (B, chunk_len, action_dim) 予測 x_1 (autograd 対応、pred v_pred 経由)
    """
    B = target_action.shape[0]
    device = target_action.device

    t = torch.rand(B, device=device)
    x_0 = torch.randn_like(target_action)
    x_1 = target_action
    t_expanded = t.view(B, 1, 1)
    x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
    v_target = x_1 - x_0

    v_pred = model(x_t, t, context, context_mask)
    loss = masked_mse(v_pred, v_target, row_mask)
    x_1_hat = x_t + v_pred * (1.0 - t_expanded)
    return loss, x_1_hat


@torch.no_grad()
def sample_action(
    model: FlowMatchingDiT,
    context: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    n_steps: int = 6,
    *,
    prefix: torch.Tensor | None = None,
    velocity_strength: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler integration for x_0 (noise) → x_1 (action)。

    - x_0 ~ N(0, 1)、shape (B, chunk_len, action_dim)
    - dt = 1 / n_steps
    - for step in range(n_steps): x += dt * model(x, step*dt, context)
    - returns x at t=1

    # RTC (Issue #137、inference 専用)

    `prefix` / `velocity_strength` は Real-Time Chunking 用の optional 引数。
    **両方 None (既定) なら従来と完全に同一** なので学習経路には影響しない。

    - `prefix`: 前 chunk の未実行分 (model action space)。ノイズの代わりに
      先頭 rows 行の初期値として置く (warm start)。
    - `velocity_strength`: (chunk_len,) の step 別スケール。速度場に掛けて
      「先頭は凍結、その先は前 chunk から徐々に離す」を実現する。生成は
      `inference/desktop/lower_policy/rtc.py: build_velocity_strength()`。

    本家 GR00T (`lerobot/policies/groot/groot_n1_7.py`) の RTC 実装と同じ形式で、
    GR00T と RAMEN-Ori で同じ YAML 設定が同じ意味になるよう揃えてある。

    Args:
        prefix: (rows, action_dim) or (B, rows, action_dim)、rows <= chunk_len。
        velocity_strength: (chunk_len,)。

    Raises:
        ValueError: prefix / velocity_strength の shape 不整合。
    """
    B = context.shape[0]
    device = context.device
    x = torch.randn(B, model.chunk_len, model.action_dim, device=device)

    if prefix is not None:
        prefix_t = prefix if prefix.dim() == 3 else prefix.unsqueeze(0)
        if prefix_t.dim() != 3 or prefix_t.shape[-1] != model.action_dim:
            raise ValueError(
                f"prefix must be (rows, {model.action_dim}) or "
                f"(B, rows, {model.action_dim}), got {tuple(prefix.shape)}"
            )
        rows = prefix_t.shape[1]
        if not 1 <= rows <= model.chunk_len:
            raise ValueError(
                f"prefix rows must be in [1, {model.chunk_len}], got {rows}"
            )
        x[:, :rows] = prefix_t.to(device=device, dtype=x.dtype)

    strength = None
    if velocity_strength is not None:
        if velocity_strength.dim() != 1 or velocity_strength.shape[0] != model.chunk_len:
            raise ValueError(
                f"velocity_strength must be ({model.chunk_len},), got "
                f"{tuple(velocity_strength.shape)}"
            )
        strength = velocity_strength.to(device=device, dtype=x.dtype).view(1, -1, 1)

    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((B,), step * dt, device=device)
        v = model(x, t, context, context_mask)
        # strength is None のときは乗算自体を行わない (既定経路を bit 一致に保つ)
        x = x + dt * v if strength is None else x + dt * v * strength
    return x


class _MeanFlowBlock(nn.Module):
    """MeanFlow 用の軽量 block: self-attn + cross-attn + MLP、AdaLN-Zero 無し、pre-LN。

    DiTBlock との差:
    - AdaLN modulation を skip (time embedding は入力 addition のみ)
    - pre-LN で通常の transformer 形式、gate 無し (単純化)
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.norm2(x)
        cross_out, _ = self.cross_attn(
            h, context, context,
            key_padding_mask=context_key_padding_mask, need_weights=False,
        )
        x = x + cross_out
        x = x + self.mlp(self.norm3(x))
        return x


class MeanFlow(nn.Module):
    """B-8 MeanFlow action expert (Issue #120 Alt-7)。

    Flow Matching の simplified variant。**straight/rectified flow** で
    velocity は constant (x_1 - x_0)、time modulation は入力 addition のみ、
    AdaLN modulation は skip、depth 半分程度 → design doc §4.4 の ~50M target。

    # Loss

    - x_0 ~ N(0, 1)、x_1 = target_action、x_t = (1-t)*x_0 + t*x_1
    - v_target = x_1 - x_0 (constant along t、"mean" flow)
    - v_pred = model(x_t, t, context)
    - loss = MSE(v_pred, v_target)  (Flow Matching と同 loss、architecture が単純)

    # Inference

    - single-step default (n_steps=1)、n_steps>1 なら Euler で複数 step
    - x_1 = x_0 + v_pred(x_0, t=0)

    # Architecture

    - action_proj + pos_embed + time_embed (sinusoidal → MLP、DiT と同 TimeEmbedding 再利用)
    - depth = 6 default (FlowMatching DiT の半分)、embed_dim = 512
    - AdaLN は使わず、time embedding は入力 addition のみ
    """

    def __init__(
        self,
        chunk_len: int = 16,
        action_dim: int = 19,
        embed_dim: int = 512,       # DiT の 768 より小
        depth: int = 6,             # DiT の 12 の半分
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        context_dim: int = 512,
        freq_dim: int = 256,
    ) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.context_dim = context_dim

        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, chunk_len, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.time_embed = TimeEmbedding(embed_dim, freq_dim=freq_dim)
        self.context_proj = nn.Linear(context_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [_MeanFlowBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm_final = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, action_dim)
        # zero-init: v_pred = 0 at init、学習安定化 (DiT と同 pattern)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # time embedding を入力 sequence に broadcast addition (AdaLN 無し simplify)
        h = self.action_proj(x) + self.pos_embed  # (B, chunk_len, embed_dim)
        t_e = self.time_embed(t).unsqueeze(1)     # (B, 1, embed_dim)
        h = h + t_e                                # time modulation
        ctx = self.context_proj(context)
        kpm = ~context_mask if context_mask is not None else None
        for block in self.blocks:
            h = block(h, ctx, kpm)
        h = self.norm_final(h)
        return self.output_proj(h)

    def compute_loss(
        self,
        target_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        row_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flow Matching と同 loss、architecture が単純化されただけ (row_mask の行だけで平均)。"""
        B = target_action.shape[0]
        device = target_action.device
        t = torch.rand(B, device=device)
        x_0 = torch.randn_like(target_action)
        x_1 = target_action
        x_t = (1 - t.view(B, 1, 1)) * x_0 + t.view(B, 1, 1) * x_1
        v_target = x_1 - x_0
        v_pred = self(x_t, t, context, context_mask)
        return masked_mse(v_pred, v_target, row_mask)

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """Straight flow: single-step default (n_steps=1)、複数 step も Euler で対応。"""
        B = context.shape[0]
        device = context.device
        x = torch.randn(B, self.chunk_len, self.action_dim, device=device)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device)
            v = self(x, t, context, context_mask)
            x = x + dt * v
        return x
