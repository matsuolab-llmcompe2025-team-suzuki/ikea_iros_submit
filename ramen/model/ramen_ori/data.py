"""RAMEN-Ori dummy dataset (Issue #115、Phase 1 M9)。

Curated subtask dataset の到着待ちで RTX 5090 smoke を回すための、決定的ランダム
tensor を返す Dataset。model.py の batch format contract に合わせて 1 sample を
dict で返し、default collate で自動 batch 化される。

# Phase 2 で切替

Real LeRobot v3 dataset は別 file `data_lerobot.py` に実装予定 (dummy は smoke debug
用途で残置)。切替は train.py の Hydra config で `_target_` を差し替えるだけ。

# Batch key (RamenOriPolicy._encode / compute_loss と同じ contract)

- images / images_prev:  (N_cams, 3, img_size, img_size)  float
- cam_id:                (N_cams,)                        long
- obb_verts:             (N_cams, top_K, 8)               float [0,1]
- obb_conf:              (N_cams, top_K, 1)               float [0,1]
- obb_class_id:          (N_cams, top_K)                  long
- obb_cam_id:            (N_cams, top_K)                  long
- obb_valid_mask:        (N_cams, top_K)                  bool
- state:                 (state_dim,)                     float
- skill_id:              scalar                            long
- action:                (chunk_len, action_dim)          float

# 決定性

`__getitem__(idx)` は idx から Generator を seed する → 同 idx は同 tensor、
smoke train の loss curve が再現可能。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DummyRamenOriDataset(Dataset):
    # 常に images_prev を返す (画像差分ありの model 用、train.py の起動時の確認で使う)
    load_prev_image = True

    def __init__(
        self,
        num_samples: int = 1000,
        # Issue #129 Phase A (2026-08-31): defaults を real recipe (3 cam / vocab 4 /
        # action 16D) に整合。旧値は N_cams=4 / num_cams=6 / action_dim=19。
        N_cams: int = 3,
        top_K: int = 4,
        num_classes: int = 7,
        num_cams: int = 4,
        num_skills: int = 6,
        state_dim: int = 71,
        chunk_len: int = 16,
        action_dim: int = 16,
        img_size: int = 224,
        augmentation_cfg: dict | None = None,   # Issue #122: dummy は aug 対象外、received but ignored
    ) -> None:
        self.num_samples = num_samples
        self.N_cams = N_cams
        self.top_K = top_K
        self.num_classes = num_classes
        self.num_cams = num_cams
        self.num_skills = num_skills
        self.state_dim = state_dim
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.img_size = img_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        gen = torch.Generator().manual_seed(idx)
        N_cams = self.N_cams
        top_K = self.top_K
        # cam_id は "cam 0..N_cams-1" の global index を default vocab 順に (data.py 側で
        # 実 dataset のカメラ名 → global id をマップする際に上書き)
        cam_ids_1d = torch.arange(N_cams, dtype=torch.long)
        # obb_cam_id: 各カメラの top_K det すべて同 cam_id を持つ
        cam_ids_2d = cam_ids_1d.unsqueeze(1).expand(N_cams, top_K).contiguous()

        return {
            "images": torch.randn(
                N_cams, 3, self.img_size, self.img_size, generator=gen
            ),
            "images_prev": torch.randn(
                N_cams, 3, self.img_size, self.img_size, generator=gen
            ),
            "cam_id": cam_ids_1d,
            "obb_verts": torch.rand(N_cams, top_K, 8, generator=gen),
            "obb_conf": torch.rand(N_cams, top_K, 1, generator=gen),
            "obb_class_id": torch.randint(
                0, self.num_classes, (N_cams, top_K), generator=gen, dtype=torch.long
            ),
            "obb_cam_id": cam_ids_2d,
            "obb_valid_mask": torch.rand(N_cams, top_K, generator=gen) > 0.3,
            "state": torch.randn(self.state_dim, generator=gen),
            "skill_id": torch.tensor(idx % self.num_skills, dtype=torch.long),
            "action": torch.randn(self.chunk_len, self.action_dim, generator=gen),
            # Issue #141 RO-2: 区間末尾の埋め草の行 (dummy は全行が実データ)
            "action_is_pad": torch.zeros(self.chunk_len, dtype=torch.bool),
        }
