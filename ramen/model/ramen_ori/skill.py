"""F-3 Skill embedding module (RAMEN-Ori、Issue #115)。

skill_id (long int) → learned embedding → 1 token を Fusion transformer に流す。
`nn.Embedding(num_skills, d_model)` の薄い wrapper、init は N(0, init_std) で
GPT/BERT/DiT 系の embedding scale に揃える (default init_std=0.02)。

# num_skills の選択 (Phase 1 では config で切替、A/B 検証用)

Dataset の canonical skill list は 2 通りある:

- **inference aligned (6 vocab)**: `inference/desktop/orchestrator.py` の 6 skill graph に
  合わせて、curation の task_index 5 (rotate_table_base) と 7 (move_table_base) を
  "move_table_base" に merge。design doc §4.3 F-3 default。
- **curation aligned (7 vocab)**: `data/curation_tool/ui/segments.py` の 7 skill 全部を
  維持 (task_index 0..5 + 7 の 7 個)。curation 側で敢えて分けた motion の差を
  学習時に保持したい場合。

どちらを選ぶかは `num_skills` (config) で切替。task_index → embedding index の mapping は
data.py 側の label mapping で吸収 (本 module は skill 名を知らない)。

# Shape

- input: skill_id (B,) long tensor、値 ∈ [0, num_skills)
- output: (B, 1, d_model)、Fusion に concat する 1 token
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SkillEmbedding(nn.Module):
    def __init__(self, num_skills: int, d_model: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.num_skills = num_skills
        self.d_model = d_model
        self.embedding = nn.Embedding(num_skills, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(self, skill_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(skill_id).unsqueeze(1)
