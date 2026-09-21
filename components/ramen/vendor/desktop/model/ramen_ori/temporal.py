"""D-3 Frame delta encoder module (RAMEN-Ori、Issue #115)。

Δ_t = I_t - I_{t-1} を small ViT (~14-20M) で encode、cam-list 可変 config + cam_id
embedding (OBB path と同 vocab を共有) で Fusion に流す motion cue token を生成。

# Shape

- input:
    - images:      (B, N_cams, 3, H, W) — I_t、normalized (ImageNet mean/std)
    - images_prev: (B, N_cams, 3, H, W) — I_{t-1}、同上
    - cam_id:      (B, N_cams) long — cam vocabulary global index (OBB path と共通)
- output: tokens (B, N_cams * pool_size**2, d_model)、default 4 cam × 64 = 256 delta token

# Design

- DeltaViT: 標準 ViT を torch primitive (nn.Conv2d + nn.TransformerEncoder) で from-scratch
  実装。外部 dep 追加なし。embed_dim=384, depth=8, heads=6, mlp_ratio=4 で ~14M param
  (design doc "small ViT ~20M" range)、pre-LN 型 (deep 側でも安定)
- TemporalEncoder wrapper: 4 cam を batch axis に統合して shared DeltaViT に流す →
  spatial pool (14×14 → 8×8) → Linear で d_model project → cam_embed を token に add
- cam_embed: OBB path と同じく `nn.Embedding(num_cams, cam_embed_dim)`、Linear で
  d_model に lift してから各 token 位置に add (standard positional-encoding pattern)
- 初期 frame (I_{-1} 無い) の扱いは data.py 責務、I_{-1} = zeros を渡す前提
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeltaViT(nn.Module):
    """Small ViT for delta image (I_t - I_{t-1}) encoding。

    構造: patch embed (Conv2d) + learned pos embed + N-layer TransformerEncoder (pre-LN、GELU)
    output shape: (B, num_patches, embed_dim)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: pre-LN (norm_first=True) と nested tensor optimization は
        # 非互換、明示的に off にして "nested tensor 使わない" warning を suppress
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, delta_img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(delta_img)  # (B, embed_dim, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        return x


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        num_cams: int = 6,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        pool_size: int = 8,
        cam_embed_dim: int = 16,
        d_model: int = 512,
    ) -> None:
        super().__init__()
        self.num_cams = num_cams
        self.pool_size = pool_size
        self.d_model = d_model
        self.embed_dim = embed_dim

        self.delta_encoder = DeltaViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.projection = nn.Linear(embed_dim, d_model)

        self.cam_embed = nn.Embedding(num_cams, cam_embed_dim)
        nn.init.normal_(self.cam_embed.weight, std=0.02)
        self.cam_lift = nn.Linear(cam_embed_dim, d_model)

    def forward(
        self,
        images: torch.Tensor,
        images_prev: torch.Tensor,
        cam_id: torch.Tensor,
    ) -> torch.Tensor:
        delta = images - images_prev  # (B, N_cams, 3, H, W)
        B, N_cams, C, H, W = delta.shape
        flat = delta.view(B * N_cams, C, H, W)

        tokens = self.delta_encoder(flat)  # (B*N_cams, num_patches, embed_dim)

        ps = self.delta_encoder.patch_size
        h, w = H // ps, W // ps
        spatial = tokens.transpose(1, 2).reshape(B * N_cams, self.embed_dim, h, w)
        pooled = self.pool(spatial)  # (B*N_cams, embed_dim, pool_size, pool_size)
        pooled = pooled.flatten(2).transpose(1, 2)  # (B*N_cams, pool_size**2, embed_dim)

        projected = self.projection(pooled)  # (B*N_cams, pool_size**2, d_model)

        cam_e = self.cam_embed(cam_id)  # (B, N_cams, cam_embed_dim)
        cam_e = self.cam_lift(cam_e)  # (B, N_cams, d_model)
        cam_e = cam_e.unsqueeze(2).expand(-1, -1, self.pool_size**2, -1)
        cam_e = cam_e.reshape(B * N_cams, self.pool_size**2, self.d_model)

        tokens_with_cam = projected + cam_e
        return tokens_with_cam.view(B, N_cams * self.pool_size**2, self.d_model)
