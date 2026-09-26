"""C-2 OBB coord token module (RAMEN-Ori、Issue #115)。

YOLO-OBB detection (per-camera top_K) を per-detection 1 token に project、Fusion に流す。
Object grounding (workspace / leg / hole 等の位置) を明示 token として与えることで、
VLA を "画像 patch から自力で物体位置認識" だけに頼らせない。design doc §4.3 C-2。

# Input format

data.py が sidecar parquet から batched tensor で流す前提:

- verts:      (B, N_cams, top_K, 8)  float, normalized xyxyxyxy [0..1]
- conf:       (B, N_cams, top_K, 1)  float [0..1]
- class_id:   (B, N_cams, top_K)     long  [0..num_classes)
- cam_id:     (B, N_cams, top_K)     long  [0..num_cams)、global cam vocabulary index
- valid_mask: (B, N_cams, top_K)     bool  detection 有効 (top_K > 実検出数 の pad は False)

`cam_id` は明示 input (軸位置から derive しない)。config `cams: [head_left, wrist_left]` の
ような subset 選択でも "cam 0 は常に head_left" が保たれる。data.py 側で cam name → global
cam_id を map。padded slot の class_id/cam_id は valid index (default 0) で埋める必要あり
(embedding OOB 回避)。

# Output

- tokens: (B, N_cams * top_K, d_model)、Fusion への flatten 済 token 列
- mask:   (B, N_cams * top_K) bool、Fusion attention mask (True = attend、False = 無視)

Padded detection の token 値は Fusion 側で attention mask により無視される前提。本 module
では零掛け等の explicit zero-out は行わない (standard transformer pattern)。

# Design

per-detection feature を cat して Linear で d_model に project:

    feature = cat(verts(8), conf(1), class_embed(class_id, 32), cam_embed(cam_id, 16)) = 57 dim
    token   = Linear(57, d_model)(feature)

Detection dropout (design doc §4.3 first candidate 5%): training mode のみ、valid_mask を
per-detection で確率的に False 反転する。dataset の "検出漏れ" を模擬 → robustness 向上。

Sort by (class_id, -confidence) は data.py 側 (offline precompute で 1 回) で行う前提、
本 module は "既に sorted" 入力として扱う。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObbTokenizer(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        num_cams: int = 6,
        class_embed_dim: int = 32,
        cam_embed_dim: int = 16,
        d_model: int = 512,
        drop_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_cams = num_cams
        self.class_embed_dim = class_embed_dim
        self.cam_embed_dim = cam_embed_dim
        self.d_model = d_model
        self.drop_rate = drop_rate

        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.cam_embed = nn.Embedding(num_cams, cam_embed_dim)
        nn.init.normal_(self.class_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cam_embed.weight, mean=0.0, std=0.02)

        # feature = verts(8) + conf(1) + class_embed + cam_embed
        feature_dim = 8 + 1 + class_embed_dim + cam_embed_dim
        self.feature_dim = feature_dim
        self.projection = nn.Linear(feature_dim, d_model)

    def forward(
        self,
        verts: torch.Tensor,
        conf: torch.Tensor,
        class_id: torch.Tensor,
        cam_id: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_e = self.class_embed(class_id)  # (B, N_cams, top_K, class_embed_dim)
        cam_e = self.cam_embed(cam_id)        # (B, N_cams, top_K, cam_embed_dim)
        features = torch.cat([verts, conf, class_e, cam_e], dim=-1)  # (..., feature_dim)
        tokens = self.projection(features)     # (B, N_cams, top_K, d_model)
        tokens = tokens.flatten(1, 2)          # (B, N_cams * top_K, d_model)
        mask = valid_mask.flatten(1, 2)        # (B, N_cams * top_K)

        if self.training and self.drop_rate > 0.0:
            keep = torch.rand(mask.shape, device=mask.device) > self.drop_rate
            mask = mask & keep

        return tokens, mask
