"""A-1 LingBot-B Vision encoder module (RAMEN-Ori、Issue #115)。

LingBot-Vision ViT-B/16 backbone (`robbyant/lingbot-vision-vit-base`、86M) を
frozen で load、4 cam RGB → per-cam patch tokens → 8×8 spatial pool →
Linear で d_model に project、Fusion に流す。

# Shape

- input: images (B, N_cams, 3, H, W)、ImageNet mean/std で normalized
  (normalize は data.py 側の transform で行う)
- output: tokens (B, N_cams * pool_size**2, d_model)、default 4 cam × 8×8 = 256 vision token

# Design

- Backbone は dependency injection (constructor で受ける) + `from_lingbot` classmethod で
  実 LingBot を load するファクトリを提供。test では stub backbone を注入 → HF DL 不要で高速
- Frozen (Phase 1 default、design doc §4.3 A-1)。`train()` override で frozen 時は backbone を
  常に eval に固定 (BN/dropout 挙動を安定化、PyTorch 慣例)
- Patch pool: 14×14 → 8×8 の `AdaptiveAvgPool2d` (design doc §4.3 A-1)。Phase 2 で
  learnable strided conv or attention pooling 検討
- Precision: Phase 1 は fp32 全一貫 (backbone forward の gradient は無いので精度気にせず)。
  Phase 2 で backbone bf16 + projection fp32 検討 (frozen 部分の速度・memory 最適化)

# LingBot backbone API 前提 (2026-07 SHA 151e463 時点)

`backbone(x, is_training=True)` が dict を返し、`x_norm_patchtokens` キーで
shape `(B, h*w, embed_dim)` の patch token tensor を持つ。`backbone.patch_size` 属性で
patch サイズ (16) を取得可能。
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        d_model: int = 512,
        pool_size: int = 8,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.pool_size = pool_size
        self.freeze = freeze

        # freeze=True: 凍結 + no_grad (土台)。freeze=False: 勾配を流し、学習する範囲は optimizer 側
        # (partial-train、最後の N block) が requires_grad で決める。LingBot の loader は凍結して返すので、
        # freeze=False だけでは学習されない (Issue #141)。
        # backbone はどちらでも常に eval mode (train() を参照)。
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone.eval()

        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.projection = nn.Linear(embed_dim, d_model)

    def train(self, mode: bool = True) -> "VisionEncoder":
        super().train(mode)
        # backbone は学習中も常に eval mode (Issue #141)。LingBot の RoPE は train mode でだけ位置の座標を
        # ランダムに拡大・縮小する (rescale_coords) ので、train mode だと学習と推論 (eval) で特徴がずれる
        # (Phase K はこの状態だった)。LingBot に dropout / drop path は無いので、eval でも partial-train の
        # 学習は変わらない。
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N_cams, C, H, W = images.shape
        flat = images.view(B * N_cams, C, H, W)

        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with ctx:
            out = self.backbone(flat, is_training=True)
        patch_tokens = out["x_norm_patchtokens"]  # (B*N_cams, h*w, embed_dim)

        ps = self.backbone.patch_size
        h, w = H // ps, W // ps

        spatial = patch_tokens.transpose(1, 2).reshape(B * N_cams, self.embed_dim, h, w)
        pooled = self.pool(spatial)  # (B*N_cams, embed_dim, pool_size, pool_size)
        pooled = pooled.flatten(2).transpose(1, 2)  # (B*N_cams, pool_size**2, embed_dim)

        tokens = self.projection(pooled)  # (B*N_cams, pool_size**2, d_model)
        return tokens.view(B, N_cams * self.pool_size**2, self.d_model)

    @classmethod
    def from_lingbot(
        cls,
        variant: str = "base",
        d_model: int = 512,
        pool_size: int = 8,
        freeze: bool = True,
        device: str | None = None,
    ) -> "VisionEncoder":
        # lazy: lingbot_vision は heavy import + HF DL trigger、from_lingbot 呼び出し時のみ load
        from lingbot_vision import load_pretrained_backbone

        # LingBot dtype vocab: "bf16" / "fp16" / "fp32" / "auto"。Phase 1 は fp32 全一貫、
        # Phase 2 で bf16 化検討 (frozen backbone の速度・memory 最適化)。
        backbone, embed_dim = load_pretrained_backbone(
            variant=variant, device=device, dtype="fp32"
        )
        instance = cls(
            backbone=backbone,
            embed_dim=embed_dim,
            d_model=d_model,
            pool_size=pool_size,
            freeze=freeze,
        )
        # projection Linear は backbone と同 device に揃える (LingBot は backbone だけ device
        # 指定を受けるので、instance.to() 呼ばないと projection が CPU 残り)
        if device is not None:
            instance.projection.to(device)
        return instance
