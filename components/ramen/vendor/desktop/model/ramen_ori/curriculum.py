"""H-3 Skill curriculum sampler (RAMEN-Ori、Issue #120 Alt-5)。

design doc §7 の Curriculum step 1-3 に沿って、step 数に応じて active skill 集合を
切替、易しい skill から順に投入。Insertion のような contact-rich skill を最初から
joint train すると崩壊しやすいので、まず move/rotate で backbone を安定 →
insertion 追加、の順に持ち込む。

# 責務分離

- **H-3 (curriculum)**: 本 module。step ベースで active skills を gate
- **H-4 (joint multi-task)**: skill embedding + 統合 model で既に実装済 (Issue #115)
- **A-5 balancing (task_uniform)**: curriculum enabled 時は本 sampler が active skill 内で
  uniform sampling するので統合、curriculum disabled 時は既存 A-5 sampler (train.py 側) を使う

# 使い方

```python
from model.ramen_ori.curriculum import CurriculumStage, CurriculumSampler

stages = [
    CurriculumStage(step=0, skills=[5]),        # move_table_base のみ
    CurriculumStage(step=3000, skills=[4, 5]),  # + rotate_table_base
]
step_holder = {"v": 0}
sampler = CurriculumSampler(
    skill_ids_per_sample=skill_ids_arr,  # shape (N,) 各 sample の skill_id
    stages=stages,
    step_ref=lambda: step_holder["v"],
    num_samples_per_epoch=len(dataset),
)
loader = DataLoader(dataset, sampler=sampler, ...)
# train loop: step_holder["v"] = step で最新 step を書き込むと、次 epoch 開始時に反映
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.utils.data


@dataclass
class CurriculumStage:
    """1 curriculum stage の定義。

    Attributes:
        step: このステージが有効になる開始 step (inclusive)
        skills: このステージで active な skill_id list (0..num_skills-1)
    """

    step: int
    skills: list[int]

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"stage.step must be >=0, got {self.step}")
        if not self.skills:
            raise ValueError("stage.skills must be non-empty")


def resolve_active_skills(
    current_step: int, stages: list[CurriculumStage]
) -> set[int]:
    """current_step を含む最新 stage の skills 集合を返す。

    stages は step 昇順であること前提 (呼び出し側 or sampler init で sort 済)。
    current_step < stages[0].step の場合は stages[0] を採用 (Phase 0 相当)。
    """
    if not stages:
        raise ValueError("stages must be non-empty")
    active = set(stages[0].skills)
    for stage in stages:
        if current_step >= stage.step:
            active = set(stage.skills)
        else:
            break
    return active


def compute_stage_weights(
    skill_ids: np.ndarray, active_skills: set[int]
) -> np.ndarray:
    """active skill に含まれる sample に uniform weight、他は 0。

    active skill 内で per-skill uniform (task_uniform) を実現するため、
    active 内の各 skill の frame 数逆比で weight を割り当てる。

    Args:
        skill_ids: (N,) 各 sample の skill_id
        active_skills: active skill_id 集合

    Returns:
        (N,) float64 weight array、sum(weights of active samples) が各 skill で等しい
    """
    n = len(skill_ids)
    weights = np.zeros(n, dtype=np.float64)
    active_list = sorted(active_skills)
    n_active = len(active_list)
    if n_active == 0:
        raise ValueError("active_skills is empty")
    for skill in active_list:
        mask = skill_ids == skill
        n_samples = int(mask.sum())
        if n_samples == 0:
            # active だが sample 無し → skip (weight 0 のまま)、ただし警告として例外
            raise ValueError(
                f"active skill_id={skill} has 0 samples in dataset — cannot sample this skill"
            )
        # per-skill 合計 = 1/n_active (curriculum の全 active skill が等頻度)
        weights[mask] = 1.0 / (n_active * n_samples)
    return weights


class CurriculumSampler(torch.utils.data.Sampler[int]):
    """H-3 Skill curriculum + H-4 joint (active 内 uniform balance) の統合 sampler。

    `__iter__` 開始時 (= epoch 開始時) に `step_ref()` を読んで active skills を
    resolve、その stage の weight で multinomial sampling。mid-epoch でのステージ
    切替は反映されない (stage 間隔 >> 1 epoch 想定)。
    """

    def __init__(
        self,
        skill_ids_per_sample: np.ndarray,
        stages: list[CurriculumStage],
        step_ref: Callable[[], int],
        num_samples_per_epoch: int | None = None,
        generator: torch.Generator | None = None,
        advantage_weights: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            advantage_weights: (N,) per-sample AWR weight (Alt-6 H-7 twist)。None なら
                curriculum stage weights のみ。stage weight と要素毎積で multinomial 確率を bias。
                mean=1 に正規化されているのが前提 (awr.compute_duration_advantage の出力)。
        """
        if not stages:
            raise ValueError("stages must be non-empty")
        # step 昇順で sort (validation を兼ねる)
        self.stages = sorted(stages, key=lambda s: s.step)
        self.skill_ids = np.asarray(skill_ids_per_sample, dtype=np.int64)
        self.step_ref = step_ref
        self.num_samples_per_epoch = (
            num_samples_per_epoch
            if num_samples_per_epoch is not None
            else len(self.skill_ids)
        )
        self.generator = generator
        if advantage_weights is not None:
            adv = np.asarray(advantage_weights, dtype=np.float64)
            if adv.shape != self.skill_ids.shape:
                raise ValueError(
                    f"advantage_weights shape {adv.shape} != skill_ids {self.skill_ids.shape}"
                )
            if (adv < 0).any():
                raise ValueError("advantage_weights must be non-negative")
            self.advantage_weights = adv
        else:
            self.advantage_weights = None

    def __len__(self) -> int:
        return self.num_samples_per_epoch

    def __iter__(self):
        current_step = int(self.step_ref())
        active = resolve_active_skills(current_step, self.stages)
        weights = compute_stage_weights(self.skill_ids, active)
        if self.advantage_weights is not None:
            # per-frame advantage を stage weight に要素毎積 (active 外は weight=0 のまま)
            weights = weights * self.advantage_weights
            total = weights.sum()
            if total == 0:
                raise ValueError(
                    "combined stage+advantage weights sum to 0 — advantage_weights "
                    "may be zero for all active-skill samples"
                )
            weights = weights / total  # multinomial は sum=1 前提でなくても動くが明示化
        indices = torch.multinomial(
            torch.from_numpy(weights),
            self.num_samples_per_epoch,
            replacement=True,
            generator=self.generator,
        )
        return iter(indices.tolist())
