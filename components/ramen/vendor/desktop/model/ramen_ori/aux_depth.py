"""I-6 Depth prediction aux head (RAMEN-Ori、Issue #120 Alt-3)。

Vision backbone (LingBot/RADIO) の feature に depth 予測 aux loss を掛けて、backbone を
"空間認識" 方向に regularize。contact-rich task (insertion, pick 押付) の精度向上狙い
(design doc §4.1 I-6、SmolVLA / LingBot-VLA の実績)。

# Architecture

VisionEncoder は (B, N_cams * pool_size**2, d_model) を返す (design doc §4.3 A-1)。
これを per-cam に reshape → conv upsample 2 段で spatial 拡大 → 1-channel log-depth
map を出力。

```
input:  (B, N_cams * pool_size^2, d_model)
    ↓ reshape to (B*N_cams, d_model, pool_size, pool_size)   e.g. (B*4, 512, 8, 8)
Conv2d(d_model, hidden, 3, padding=1) + GELU                  e.g. 512→256
    ↓ Upsample 8→16
Conv2d(hidden, hidden//2, 3, padding=1) + GELU                256→128
    ↓ Upsample 16→32
Conv2d(hidden//2, 1, 3, padding=1)                            128→1
output: (B, N_cams, 1, H_out, W_out)                          e.g. (B, 4, 1, 32, 32)
```

# Loss integration (model.py 側)

`RamenOriPolicy.compute_loss` で `aux_head is not None and "depth_target" in batch` の
時のみ aux loss を加算:

    total_loss = action_loss + aux_weight * L1(depth_pred, depth_target)

`depth_target_mask` が batch にあれば、有効 pixel のみ loss 計算 (SGBM valid mask 用)。

# Target format

- log-depth 推奨 (numerical stability、Eigen 2014)。ただし本 head は raw を返すので
  target 側で log 化するか、L1 loss なら raw のままでも学習可
- shape: (B, N_cams, 1, H_out, W_out) と一致要
- SGBM+WLS pipeline (Phase 0 precompute item、別 Alt) 未実装時は data 側で zeros
  placeholder、`data.load_depth_target=false` で aux loss skip 相当

# Param 見積 (default: pool_size=8, output_size=(32,32), d_model=512, hidden=256)

- Conv1 (512→256, 3x3): 512*256*9 + 256 ≈ 1.18M
- Conv2 (256→128, 3x3): 256*128*9 + 128 ≈ 295K
- Conv3 (128→1, 3x3):   128*1*9 + 1     ≈ 1.15K
- 合計 ~1.5M (design doc §4.4 の "~5M" 見積より軽量、hidden UP で 5M 到達可能)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthHead(nn.Module):
    def __init__(
        self,
        vision_embed_dim: int = 512,
        num_cams: int = 4,
        pool_size: int = 8,           # VisionEncoder pool_size と同じ
        output_size: tuple[int, int] = (32, 32),
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.vision_embed_dim = vision_embed_dim
        self.num_cams = num_cams
        self.pool_size = pool_size
        self.output_size = tuple(output_size)
        self.hidden_dim = hidden_dim

        # spatial upsample 段数 = log2(output_size / pool_size)、2 段 default (8→16→32)
        H_out, W_out = self.output_size
        if H_out != W_out:
            raise ValueError(f"square output only、got {output_size}")
        if H_out % pool_size != 0:
            raise ValueError(f"output_size ({H_out}) must be multiple of pool_size ({pool_size})")
        n_upsamples = 0
        s = pool_size
        while s < H_out:
            s *= 2
            n_upsamples += 1
        if s != H_out:
            raise ValueError(
                f"output_size ({H_out}) must be pool_size ({pool_size}) * 2^k"
            )

        # decoder: Conv → GELU → Upsample を n_upsamples 回、最後に 1-ch conv
        layers: list[nn.Module] = []
        in_ch = vision_embed_dim
        cur_hidden = hidden_dim
        for i in range(n_upsamples):
            out_ch = cur_hidden if i == 0 else cur_hidden // 2
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.GELU())
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            in_ch = out_ch
            cur_hidden = out_ch
        # 最終 1-ch conv (upsample 後、H_out spatial で予測)
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, N_cams * pool_size^2, d_model) VisionEncoder 出力

        Returns:
            (B, N_cams, 1, H_out, W_out) 予測 depth (raw、target 側で log 化 or raw)
        """
        B = vision_features.shape[0]
        expected_N = self.num_cams * self.pool_size * self.pool_size
        if vision_features.shape[1] != expected_N:
            raise ValueError(
                f"vision_features N mismatch: got {vision_features.shape[1]}, "
                f"expected N_cams({self.num_cams}) * pool_size**2({self.pool_size**2}) = {expected_N}"
            )
        if vision_features.shape[2] != self.vision_embed_dim:
            raise ValueError(
                f"vision_features d_model mismatch: got {vision_features.shape[2]}, "
                f"expected {self.vision_embed_dim}"
            )

        # (B, N_cams * P^2, d) → (B, N_cams, P, P, d) → (B, N_cams, d, P, P)
        P = self.pool_size
        x = vision_features.view(B, self.num_cams, P, P, self.vision_embed_dim)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B, N_cams, d, P, P)
        # (B * N_cams, d, P, P) で conv 適用
        x = x.view(B * self.num_cams, self.vision_embed_dim, P, P)
        out = self.decoder(x)  # (B*N_cams, 1, H_out, W_out)
        H_out, W_out = self.output_size
        return out.view(B, self.num_cams, 1, H_out, W_out)
